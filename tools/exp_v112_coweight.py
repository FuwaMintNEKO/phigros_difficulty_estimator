# -*- coding: utf-8 -*-
"""推理层权重修正: 高估主因特征降权 (密度/位移/交替/变速)"""
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

# 权重修正方案 (推理层覆盖)
CO_OVERRIDES = {
    'v112a': {  # 温和: 密度/位移/交替/变速 降权
        'above_avg_density_mean': 0.27, 'real_core_notes_per_second': 0.07,
        'movement_per_second': 0.04, 'movement_density_index': 0.04,
        'chord_alternation_rate': 0.15, 'tempo_change_count': 0.02,
        'type_switch_per_sec': 0.08,
    },
    'v112b': {  # 中等
        'above_avg_density_mean': 0.24, 'real_core_notes_per_second': 0.06,
        'movement_per_second': 0.03, 'movement_density_index': 0.03,
        'chord_alternation_rate': 0.12, 'tempo_change_count': 0.015,
        'type_switch_per_sec': 0.07,
    },
}

def predict(feats_raw, level='IN', co_ov=None):
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
    mf_scale = (0.70 if dens >= 12.5 else 0.50) if mf3 >= 30 else (1.0 if mf3 <= 5 else 0.8)
    eff_scale = 1.0 if mf3 >= 30 else (1.5 if mf3 <= 5 else 1.0)
    total = 0.0
    cd_ = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        if co_ov and fname in co_ov: co = co_ov[fname]
        v = feats.get(fname, 0)
        pv = P95.get(fname, 0)
        t = max(pv*0.55, bl*0.5)
        if v <= t: continue
        e = v/t - 1.0
        c = CAPS.get(fname, cd_)
        if c is not None and e > c: e = c
        co2 = co * (mf_scale if fname in MF_FEATS else (eff_scale if fname in EFF_FEATS else 1.0))
        x_ = co2 * (e**0.70)
        p99 = max(P99.get(fname,0), bl*0.5)
        if v > p99:
            pe = v/p99 - 1.0
            if c is not None and pe > c: pe = c
            x_ += co2*max(0,pe)**0.70*0.5
        total += x_
    return p_gb + total

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
valid = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]

for vname, co_ov in [('baseline', None), *list(CO_OVERRIDES.items())]:
    bins = {}
    for r in valid:
        d = r['diff']
        p = predict(r['feats'], r['level'], co_ov)
        bin_ = d < 13 and '<13' or d < 14 and '13-14' or d < 15 and '14-15' or d < 16 and '15-16' or d < 17 and '16-17' or '>=17'
        b = bins.setdefault(bin_, {'n':0,'b':0,'mae':0})
        b['n'] += 1; b['b'] += p-d; b['mae'] += abs(p-d)
    line = ' | '.join(f'{k}:{b["b"]/b["n"]:+.2f}/{b["mae"]/b["n"]:.2f}' for k, b in sorted(bins.items(), key=lambda x: float(x[0].replace('<','0').replace('-','.').replace('>=','99'))))
    print(f'{vname:<10} {line}')
