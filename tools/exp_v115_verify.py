# -*- coding: utf-8 -*-
"""v11.5 生产路径验证: ranked清单 + 案例 (完整生产逻辑: domain align + 条件boost + 定轨 + 校准)
用法: python tools/exp_v115_verify.py [model_path]
"""
import os, sys, pickle, numpy as np, io, json, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from scipy.stats import spearmanr
from boost_config import MANUAL_FLAT

MODEL = sys.argv[1] if len(sys.argv) > 1 else '6dim_model_v11_5.pkl'
m = pickle.load(open(os.path.join(_ROOT, 'models', MODEL), 'rb'))
gb, scaler = m['gb'], m['scaler']
FN = m['feature_names']; LV_ORDER = m['lv_order']
P95 = m['p95_vals']; P99 = m['p99_vals']
FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT); CAPS = m.get('caps', {})
_ALIGN = json.load(open(os.path.join(_ROOT, 'data', 'domain_align.json'), encoding='utf-8')).get('delta', {})

MF_FEATS_COND = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS_COND = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
DENS_FEATS_COND = {'above_avg_density_mean', 'real_core_notes_per_second'}
EXTREME_FEATS_COND = {'cross_hand_density', 'jline_relative_cross', 'thirtysecond_run_max', 'thirtysecond_run_ratio', 'lane_switch_density'}

def predict(feats_raw, level='IN', is_custom=True):
    feats = dict(feats_raw)
    lv = level.upper()
    if 'AT' in lv: lv = 'AT'
    elif 'IN' in lv: lv = 'IN'
    elif 'HD' in lv: lv = 'HD'
    elif 'EZ' in lv: lv = 'EZ'
    else: lv = 'IN'
    if is_custom and lv == 'IN':
        for k, d in _ALIGN.items():
            if k in feats: feats[k] = feats[k] - d
    if 'IN_AT' in LV_ORDER and lv in ('IN','AT'): lv = 'IN_AT'
    if lv not in LV_ORDER: lv = LV_ORDER[-1]
    vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(lv)] = 1.0
    x = np.array([[feats.get(n,0) for n in FN] + vec])
    p_gb = float(gb.predict(scaler.transform(x))[0])
    mf3 = feats_raw.get('multi_finger_3plus_events', 0); dens = feats_raw.get('above_avg_density_mean', 0)
    ml = feats_raw.get('multi_line_sim_events', 0); wmf = feats_raw.get('weighted_mf_score_per_sec', 0)
    if mf3 >= 30 and ml >= 100: mf_scale, dens_s = 0.45, 0.85
    elif mf3 >= 30: mf_scale, dens_s = (0.70 if dens >= 9.5 else 0.50), 1.0
    else: mf_scale, dens_s = (1.0 if mf3 <= 5 else 0.8), 1.0
    if mf3 <= 5:
        _sw = min(max((wmf - 12.0) / 6.0, 0.0), 1.0)
        eff_scale = 1.0 if dens >= 10.0 else 1.5 - 0.5 * _sw
        wmf_scale = 1.0 - 0.4 * _sw
        extreme_scale = 1.3
    elif mf3 >= 30:
        eff_scale, wmf_scale = 1.0, 1.0
        extreme_scale = 0.70
    else:
        eff_scale, wmf_scale = 1.0, 1.0
        extreme_scale = 1.0
    total = 0.0; cd_ = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0); pv = P95.get(fname, 0)
        t = max(pv*0.55, bl*0.5)
        if v <= t: continue
        e = v/t - 1.0
        c = CAPS.get(fname, cd_)
        if c is not None and e > c: e = c
        co2 = co
        if is_custom:
            if fname in MF_FEATS_COND: co2 = co * mf_scale
            elif fname in EFF_FEATS_COND: co2 = co * eff_scale
            if fname in DENS_FEATS_COND and mf3 >= 30 and ml >= 100: co2 = co * dens_s
            if fname == 'weighted_mf_score_per_sec': co2 = co * wmf_scale
            if fname in EXTREME_FEATS_COND: co2 = co * extreme_scale
        x_ = co2 * (e**0.70)
        p99 = max(P99.get(fname,0), bl*0.5)
        if v > p99:
            pe = v/p99 - 1.0
            if c is not None and pe > c: pe = c
            x_ += co2*max(0,pe)**0.70*0.5
        total += x_
    pred = p_gb + total
    act = feats_raw.get('tracks_active_sec', 0)
    if act > 0:
        r4 = feats_raw.get('tracks_4plus_sec', 0)/act; r5 = feats_raw.get('tracks_5plus_sec', 0)/act; r6 = feats_raw.get('tracks_6plus_sec', 0)/act
        pred += 0.15*min(r4,0.8) + 0.55*min(r5,0.4) + 1.0*min(r6,0.15)
    if is_custom:
        for lo, hi, adj in [(14,15,0.55),(15,16,0.40),(16,17,0.20)]:
            if lo < pred <= hi: pred -= adj; break
    return pred, p_gb, total

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)

# 官谱 in-sample (is_custom=False)
official = cache['official']
ds_o = np.array([r['diff'] for r in official])
ps_o = np.array([predict(r['feats'], r['level'], is_custom=False)[0] for r in official])
print(f'===== {MODEL} 官谱 in-sample =====')
print(f'MAE={np.abs(ps_o-ds_o).mean():.4f} bias={(ps_o-ds_o).mean():+.4f}')

# ranked (is_custom=True 生产路径)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([r['diff'] for r in ranked])
ps = np.array([predict(r['feats'], r['level'])[0] for r in ranked])
errs = ps - ds
rho, _ = spearmanr(ds, ps)
print(f'\n===== ranked 589 (生产路径, 含极端配置缩放) =====')
print(f'MAE={np.abs(errs).mean():.3f} bias={errs.mean():+.3f} Spearman={rho:.3f}')
for lo, hi, tag in [(0,14,'<14'),(14,15,'14-15'),(15,16,'15-16'),(16,17,'16-17'),(17,99,'>=17')]:
    mk = np.where((ds >= lo) & (ds < hi))[0]
    if len(mk): print(f'  [{tag}]: n={len(mk)} bias={errs[mk].mean():+.3f} MAE={np.abs(errs[mk]).mean():.3f}')
mf = np.array([r['feats'].get('multi_finger_3plus_events', 0) for r in ranked])
for lbl, cond in [('多指', mf>=30), ('双指', mf<=5), ('混合', (mf>5)&(mf<30))]:
    g = np.where(cond)[0]
    if len(g): print(f'  {lbl}: n={len(g)} bias={errs[g].mean():+.3f} MAE={np.abs(errs[g]).mean():.3f}')

# 案例
print('\n===== 案例 =====')
targets = {'Feeling Blue', '3rd Avenue', 'Grown-up', '寄明月', 'cyanine', 'Secret of my heart', '甜甜'}
for r in ranked:
    if any(t.lower() in r['name'].lower() for t in targets):
        pr, g_, b_ = predict(r['feats'], r['level'])
        print(f'  {r["name"][:30]:<32} 社区={r["diff"]:.1f} 预测={pr:.2f} err={pr-r["diff"]:+.2f} (gb={g_:.2f} boost={b_:.2f}) mf3={r["feats"].get("multi_finger_3plus_events",0):.0f}')

# 新特征重要性
print('\n===== 新特征 GB 重要性 =====')
imp = gb.feature_importances_
fn_all = FN + LV_ORDER
newk = ['lane_switch', 'crossline_chain', 'jline_relative_cross', 'tempo_change_log', 'speed_event_log', 'speed_volatility_log', 'thirtysecond']
for k in newk:
    hits = [(f, i) for f, i in zip(fn_all, imp) if k in f]
    for f, i in hits:
        print(f'  {i*100:5.2f}%  {f}')
print('DONE')
