# -*- coding: utf-8 -*-
"""双指谱密度分化模拟: 
双指(mf3<=5): dens>=13(官谱P90附近) → eff×1.0 + wmf×0.7; dens<13 → eff×1.5(保持)
"""
import os, sys, pickle, numpy as np, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT

with open(os.path.join(_ROOT, 'models', '6dim_model_v11_1.pkl'), 'rb') as f:
    m = pickle.load(f)
gb, scaler = m['gb'], m['scaler']
FN, P95, P99 = m['feature_names'], m['p95_vals'], m['p99_vals']
LV_ORDER = m.get('lv_order', ['EZ','HD','IN','AT'])
CAPS = m.get('caps', {})
FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT)
MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}

def predict(feats_raw, level='IN', dens_th=13.0):
    feats = dict(feats_raw)
    lv = level.upper()
    if 'IN_AT' in LV_ORDER and lv in ('IN','AT'): lv = 'IN_AT'
    if lv not in LV_ORDER: lv = LV_ORDER[-1]
    vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(lv)] = 1.0
    x = np.array([[feats.get(n,0) for n in FN] + vec])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    mf3 = feats_raw.get('multi_finger_3plus_events', 0)
    dens = feats_raw.get('above_avg_density_mean', 0)
    if mf3 >= 30:
        mf_scale = 0.70 if dens >= 12.5 else 0.50
        eff_scale = 1.0
    elif mf3 <= 5:
        mf_scale = 1.0
        eff_scale = 1.0 if dens >= dens_th else 1.5   # 高密度双指不抬eff
        wmf_scale = 0.7 if dens >= dens_th else 1.0
    else:
        mf_scale, eff_scale, wmf_scale = 0.8, 1.0, 1.0
    total = 0.0
    cd_ = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0)
        pv = P95.get(fname, 0)
        t = max(pv*0.55, bl*0.5)
        if v <= t: continue
        e = v/t - 1.0
        c = CAPS.get(fname, cd_)
        if c is not None and e > c: e = c
        co2 = co * (mf_scale if fname in MF_FEATS else (eff_scale if fname in EFF_FEATS else 1.0))
        if fname == 'weighted_mf_score_per_sec' and mf3 <= 5: co2 = co * wmf_scale
        x_ = co2 * (e**0.70)
        p99 = max(P99.get(fname,0), bl*0.5)
        if v > p99:
            pe = v/p99 - 1.0
            if c is not None and pe > c: pe = c
            x_ += co2*max(0,pe)**0.70*0.5
        total += x_
    pred = p_gb + total
    for lo, hi, adj in [(14,15,0.30),(15,16,0.18),(16,17,0.05)]:
        if lo < pred <= hi: pred -= adj; break
    return pred

# 上架谱评估
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
valid = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]

for th in [None, 12.5, 13.0, 13.5]:
    bins = {}
    for r in valid:
        d = r['diff']
        p = predict(r['feats'], r['level'], th if th else 99.0)
        bin_ = d < 14 and '<14' or d < 15 and '14-15' or d < 16 and '15-16' or d < 17 and '16-17' or '>=17'
        b = bins.setdefault(bin_, {'n':0,'b':0})
        b['n'] += 1; b['b'] += p-d
    tag = f'th={th}' if th else 'baseline(全抬1.5)'
    line = ' | '.join(f'{k}:{b["b"]/b["n"]:+.2f}' for k, b in sorted(bins.items(), key=lambda x: float(x[0].replace('<','0').replace('-','.').replace('>=','99'))))
    print(f'{tag:<18} {line}')
    # 双指组细分
    df = [r for r in valid if r['feats'].get('multi_finger_3plus_events', 0) <= 5 and 14 <= r['diff'] < 17]
    if df and th:
        errs = [predict(r['feats'], r['level'], th) - r['diff'] for r in df]
        df_hi = [r for r in df if r['feats'].get('above_avg_density_mean', 0) >= th]
        df_lo = [r for r in df if r['feats'].get('above_avg_density_mean', 0) < th]
        eh = [predict(r['feats'], r['level'], th) - r['diff'] for r in df_hi]
        el = [predict(r['feats'], r['level'], th) - r['diff'] for r in df_lo]
        print(f'  双指14-17: n={len(df)} bias={np.mean(errs):+.3f} | 高密({len(df_hi)})={np.mean(eh):+.3f} 低密({len(df_lo)})={np.mean(el):+.3f}')
