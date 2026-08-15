# -*- coding: utf-8 -*-
"""v12 校准扫描: 用 v12 模型对 ranked(上架)预测, 贪心搜索7段校准表使MAE最小
输出: 最优7段表 + 各段bias + 总MAE (与v11.13的7段划分一致)
"""
import os, sys, io, json, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import importlib
import app as app_mod
importlib.reload(app_mod)
from scipy.stats import spearmanr

def lv_key(s):
    s = (s or '').upper()
    if 'AT' in s: return 'AT'
    if 'IN' in s: return 'IN'
    if 'HD' in s: return 'HD'
    return 'IN'

SEGS = [(12, 13), (13, 14), (14, 15), (15, 16), (16, 16.5), (16.5, 17), (17, 99)]

def predict_full(feats_raw, level_str, calib):
    feats = dict(feats_raw)
    lv = lv_key(level_str)
    if lv == 'IN':
        for k, d in app_mod.DOMAIN_DELTA.items():
            if k in feats: feats[k] = feats[k] - d
    lv2 = 'IN_AT' if lv in ('IN', 'AT') and 'IN_AT' in app_mod.LV_ORDER else lv
    if lv2 not in app_mod.LV_ORDER: lv2 = app_mod.LV_ORDER[-1]
    vec = [0.0] * len(app_mod.LV_ORDER); vec[app_mod.LV_ORDER.index(lv2)] = 1.0
    x = np.array([[feats.get(n, 0) for n in app_mod.FN] + vec])
    p_gb = float(app_mod.gb.predict(app_mod.scaler.transform(x))[0])
    b, _, _ = app_mod.compute_boost(feats, 1.0, is_custom=True)
    pred = p_gb + b
    _H = {'叠键', '多押', '变速', '位移'}
    if 14 < pred <= 16.5 and sum(1 for t in app_mod.compute_tags(feats) if t in _H) >= 2:
        pred -= b * 0.08
    act = feats.get('tracks_active_sec', 0)
    if act > 0:
        pred += 0.15 * min(feats.get('tracks_4plus_sec', 0) / act, 0.8) \
             + 0.55 * min(feats.get('tracks_5plus_sec', 0) / act, 0.4) \
             + 1.0 * min(feats.get('tracks_6plus_sec', 0) / act, 0.15)
    hr = feats.get('hold_count', 0) / max(feats.get('total_notes', 1), 1)
    if hr >= 0.6: pred += 0.7
    elif hr >= 0.4: pred += 0.5
    elif hr >= 0.25: pred += 0.3
    # v12.3/12.4: 与app.py一致的双指/多指类型偏置
    mf3 = feats.get('multi_finger_3plus_events', 0)
    dens = feats.get('above_avg_density_mean', 0)
    if mf3 <= 5 and dens >= 8.0: pred += 0.00
    elif mf3 >= 30: pred -= 0.15
    for lo, hi, adj in calib:
        if lo < pred <= hi: pred -= adj; break
    return pred

charts = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
up_ids = {c['id'] for c in charts['上架']}
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10 and r['id'] in up_ids]
ds = np.array([round(r['diff'], 1) for r in ranked])
int_mask = np.abs(ds - np.round(ds)) < 1e-6
ranked_f = [r for i, r in enumerate(ranked) if not int_mask[i]]
ds_f = ds[~int_mask]
print(f'评估集: {len(ranked_f)} 首 (上架/非整数定数/diff>10)')

def eval_calib(calib):
    ps = np.array([predict_full(r['feats'], r['level'], calib) for r in ranked_f])
    errs = ps - ds_f
    return errs, ps

# 基线(无校准)
errs0, ps0 = eval_calib([])
print(f'无校准: MAE={np.abs(errs0).mean():.3f} bias={errs0.mean():+.3f}')
for lo, hi in SEGS:
    mk = np.where((ds_f >= lo) & (ds_f < hi))[0]
    if len(mk):
        print(f'  [{lo},{hi}): n={len(mk)} bias={errs0[mk].mean():+.3f}')

# 校准搜索: bias反号初始化 + 阻尼迭代 (段间耦合自适应; 避免贪心在16.5+/17+小样本上过度校准)
best = dict((seg, 0.0) for seg in SEGS)
for lo, hi in SEGS:
    mk = np.where((ds_f >= lo) & (ds_f < hi))[0]
    if len(mk):
        best[(lo, hi)] = -float(errs0[mk].mean())  # bias反号
print(f'init: {[(lo, hi, round(best[(lo, hi)], 3)) for lo, hi in SEGS]}')
for _round in range(6):
    errs_c, _ = eval_calib([(l, h, best[(l, h)]) for l, h in SEGS])
    moved = 0.0
    for lo, hi in SEGS:
        mk = np.where((ds_f >= lo) & (ds_f < hi))[0]
        if len(mk):
            seg_bias = float(errs_c[mk].mean())
            # 阻尼: adj += bias*0.7, 限幅±0.6
            new_adj = max(-0.6, min(0.6, best[(lo, hi)] + seg_bias * 0.7))
            moved += abs(new_adj - best[(lo, hi)])
            best[(lo, hi)] = new_adj
    errs_c, _ = eval_calib([(l, h, best[(l, h)]) for l, h in SEGS])
    print(f'round{_round}: MAE={np.abs(errs_c).mean():.4f} 表={[(lo, hi, round(best[(lo, hi)], 2)) for lo, hi in SEGS]}')
    if moved < 0.005:
        break

final_calib = [(lo, hi, round(best[(lo, hi)], 2)) for lo, hi in SEGS]
errs, ps = eval_calib(final_calib)
print(f'\n===== v12 最终 =====')
print(f'MAE={np.abs(errs).mean():.3f} RMSE={np.sqrt((errs**2).mean()):.3f} bias={errs.mean():+.3f} rho={spearmanr(ps, ds_f).statistic:.3f}')
print(f'_CALIB_TABLE = {final_calib}')
for lo, hi in SEGS:
    mk = np.where((ds_f >= lo) & (ds_f < hi))[0]
    if len(mk):
        print(f'  [{lo},{hi}): n={len(mk)} bias={errs[mk].mean():+.3f} MAE={np.abs(errs[mk]).mean():.3f}')
# 特谱检查: 魔理沙(特殊分区)排除确认
sp_ids = {c['id'] for c in charts.get('特殊', [])}
n_sp = sum(1 for r in ranked if r['id'] in sp_ids)
print(f'注: 特殊分区 {n_sp} 首已排除')
