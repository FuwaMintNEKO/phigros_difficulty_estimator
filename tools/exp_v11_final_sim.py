# -*- coding: utf-8 -*-
"""最终方案模拟: 基线模型 + 条件boost(mf3衰减0.40/eff 1.50) + 段校准
用特征缓存 feats_cache_v11.pkl, 秒级完成
"""
import os, sys, pickle, numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT

with open(os.path.join(_ROOT, 'models', '6dim_model_v11.pkl'), 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']
LV_ORDER = m.get('lv_order', ['EZ','HD','IN','AT'])
CAPS = m.get('caps', {})
FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT)

MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
MF3_GE30 = 0.50   # 低密度多指谱(堆料型) mf系数
MF3_LE5 = 1.0     # 双指谱不动
MF3_MID = 0.8     # 混合
EFF_GE30 = 1.0
EFF_LE5 = 1.50    # 双指谱 eff 抬升
EFF_MID = 1.0

_ALIGN = {}
try:
    with open(os.path.join(_ROOT, 'data', 'domain_align.json'), encoding='utf-8') as _f:
        _ALIGN = json.load(_f).get('delta', {})
except Exception:
    pass
import json

def level_key(s):
    s = (s or '').upper()
    if 'AT' in s: return 'AT'
    if 'IN' in s: return 'IN'
    if 'HD' in s: return 'HD'
    return 'IN'

def level_onehot(lv):
    lv = level_key(lv)
    if 'IN_AT' in LV_ORDER and lv in ('IN','AT'): lv = 'IN_AT'
    if lv not in LV_ORDER: lv = LV_ORDER[-1]
    vec = [0.0]*len(LV_ORDER)
    vec[LV_ORDER.index(lv)] = 1.0
    return vec

def predict(feats_raw, level, do_calib=True):
    feats = dict(feats_raw)
    if level_key(level) == 'IN':
        for k, d in _ALIGN.items():
            if k in feats: feats[k] = feats[k] - d
    x = np.array([[feats.get(n,0) for n in FN] + level_onehot(level)])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    mf3 = feats_raw.get('multi_finger_3plus_events', 0)
    if mf3 >= 30:
        # 密度条件: 高密度多指谱(真材实料)少压, 低密度多指谱(堆料型)重压
        dens = feats_raw.get('above_avg_density_mean', 0)
        mf_scale = 0.70 if dens >= 12.5 else MF3_GE30
    else:
        mf_scale = MF3_LE5 if mf3 <= 5 else MF3_MID
    eff_scale = EFF_GE30 if mf3 >= 30 else (EFF_LE5 if mf3 <= 5 else EFF_MID)
    total = 0.0
    cd = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0)
        pv = P95.get(fname, 0)
        t = max(pv*0.55, bl*0.5)
        if v <= t: continue
        e = v/t - 1.0
        c = CAPS.get(fname, cd)
        if c is not None and e > c: e = c
        co2 = co
        if fname in MF_FEATS: co2 = co * mf_scale
        elif fname in EFF_FEATS: co2 = co * eff_scale
        x_ = co2 * (e**0.70)
        p99 = max(P99.get(fname,0), bl*0.5)
        if v > p99:
            pe = v/p99 - 1.0
            if c is not None and pe > c: pe = c
            x_ += co2*max(0,pe)**0.70*0.5
        total += x_
    pred = p_gb + total
    if do_calib:
        # 预测时校准（仅自制谱, 按预测值段; 基于当前生产口径诊断更新）
        if 14 < pred <= 15: pred -= 0.30
        elif 15 < pred <= 16: pred -= 0.18
        elif 16 < pred <= 17: pred -= 0.05
        # 13-14 不校(当前-0.04); >=17 不校(外推保持趋势)
    return pred

# 加载缓存
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = cache['ranked']
valid = [r for r in ranked if r['diff'] and r['diff'] > 10]
print(f'有效上架谱: {len(valid)}')

results = []
for r in valid:
    pred = predict(r['feats'], r['level'])
    results.append({'name': r['name'], 'diff': r['diff'], 'pred': pred, 'feats': r['feats']})

print('\n===== 最终方案 (条件boost + 校准) ====')
bins = {}
for r in results:
    d = r['diff']
    bin_ = d < 13 and '<13' or d < 14 and '13-14' or d < 15 and '14-15' or d < 16 and '15-16' or d < 17 and '16-17' or '>=17'
    b = bins.setdefault(bin_, {'n':0,'b':0,'mae':0})
    b['n'] += 1; b['b'] += r['pred']-d; b['mae'] += abs(r['pred']-d)
for k in sorted(bins, key=lambda x: float(x.replace('<','0').replace('-','.').replace('>=','99'))):
    b = bins[k]
    print(f'  {k}: n={b["n"]} bias={b["b"]/b["n"]:+.3f} MAE={b["mae"]/b["n"]:.3f}')
hi = [r for r in results if r['diff'] >= 16]
groups = {}
for r in hi:
    mf3 = r['feats'].get('multi_finger_3plus_events', 0)
    g = '多指(mf3>=30)' if mf3 >= 30 else ('双指(mf3<=5)' if mf3 <= 5 else '混合')
    gr = groups.setdefault(g, {'n':0,'b':0,'mae':0})
    gr['n'] += 1; gr['b'] += r['pred']-r['diff']; gr['mae'] += abs(r['pred']-r['diff'])
print('  16+ 分组:')
for g, gr in groups.items():
    print(f'    {g}: n={gr["n"]} bias={gr["b"]/gr["n"]:+.3f} MAE={gr["mae"]/gr["n"]:.3f}')

# 外推段
ext = [r for r in results if r['diff'] >= 17.7]
print('\n=== 外推段 (>=17.7) ===')
for r in sorted(ext, key=lambda x: -x['diff']):
    mf3 = r['feats'].get('multi_finger_3plus_events', 0)
    print(f'  {r["name"][:26]}: 社区 {r["diff"]:.1f} | 预测 {r["pred"]:.2f} | 差 {r["pred"]-r["diff"]:+.2f} | mf3={mf3}')
print('\nDONE')