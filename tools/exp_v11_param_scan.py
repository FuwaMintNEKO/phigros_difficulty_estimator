# -*- coding: utf-8 -*-
"""参数扫描: mf衰减力度 x 校准表 组合, 用特征缓存快速评估"""
import os, sys, pickle, json, numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT

with open(os.path.join(_ROOT, 'models', '6dim_model_v10.pkl'), 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']
LV_ORDER = m.get('lv_order', ['EZ','HD','IN','AT'])
CAPS = m.get('caps', {})
FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT)
MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
_ALIGN = json.load(open(os.path.join(_ROOT, 'data', 'domain_align.json'), encoding='utf-8')).get('delta', {})

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

def predict(feats_raw, level, mf_ge30, eff_le5, calib):
    feats = dict(feats_raw)
    if level_key(level) == 'IN':
        for k, d in _ALIGN.items():
            if k in feats: feats[k] = feats[k] - d
    x = np.array([[feats.get(n,0) for n in FN] + level_onehot(level)])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    mf3 = feats_raw.get('multi_finger_3plus_events', 0)
    mf_scale = mf_ge30 if mf3 >= 30 else (1.0 if mf3 <= 5 else 0.8)
    eff_scale = 1.0 if mf3 >= 30 else (eff_le5 if mf3 <= 5 else 1.0)
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
        co2 = co * (mf_scale if fname in MF_FEATS else (eff_scale if fname in EFF_FEATS else 1.0))
        x_ = co2 * (e**0.70)
        p99 = max(P99.get(fname,0), bl*0.5)
        if v > p99:
            pe = v/p99 - 1.0
            if c is not None and pe > c: pe = c
            x_ += co2*max(0,pe)**0.70*0.5
        total += x_
    pred = p_gb + total
    for lo, hi, adj in calib:
        if lo < pred <= hi: pred -= adj; break
    return pred

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
valid = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]

configs = [
    ('A: mf0.50 eff1.5 calib[14-15:0.25,15-16:0.18]', 0.50, 1.50, [(14,15,0.25),(15,16,0.18)]),
    ('B: mf0.50 eff1.5 calib[14-15:0.20,15-16:0.15,16-17:0.05]', 0.50, 1.50, [(14,15,0.20),(15,16,0.15),(16,17,0.05)]),
    ('C: mf0.60 eff1.5 calib[14-15:0.25,15-16:0.18]', 0.60, 1.50, [(14,15,0.25),(15,16,0.18)]),
    ('D: mf0.50 eff1.6 calib[14-15:0.25,15-16:0.18]', 0.50, 1.60, [(14,15,0.25),(15,16,0.18)]),
    ('E: mf0.45 eff1.5 calib[14-15:0.25,15-16:0.18,13-14:0.10]', 0.45, 1.50, [(13,14,0.10),(14,15,0.25),(15,16,0.18)]),
]
for cname, mf30, eff5, calib in configs:
    results = []
    for r in valid:
        results.append({'diff': r['diff'], 'pred': predict(r['feats'], r['level'], mf30, eff5, calib), 'feats': r['feats']})
    bins = {}
    for r in results:
        d = r['diff']
        bin_ = d < 13 and '<13' or d < 14 and '13-14' or d < 15 and '14-15' or d < 16 and '15-16' or d < 17 and '16-17' or '>=17'
        b = bins.setdefault(bin_, {'n':0,'b':0,'mae':0})
        b['n'] += 1; b['b'] += r['pred']-d; b['mae'] += abs(r['pred']-d)
    hi = [r for r in results if r['diff'] >= 16]
    groups = {}
    for r in hi:
        mf3 = r['feats'].get('multi_finger_3plus_events', 0)
        g = 'MF' if mf3 >= 30 else ('DF' if mf3 <= 5 else 'MX')
        gr = groups.setdefault(g, {'n':0,'b':0})
        gr['n'] += 1; gr['b'] += r['pred']-r['diff']
    line = ' | '.join(f'{k}:{b["b"]/b["n"]:+.2f}/{b["mae"]/b["n"]:.2f}' for k, b in sorted(bins.items(), key=lambda x: float(x[0].replace('<','0').replace('-','.').replace('>=','99'))))
    grline = ' | '.join(f'{g}:{gr["b"]/gr["n"]:+.2f}' for g, gr in groups.items())
    print(f'{cname}')
    print(f'  段: {line}')
    print(f'  16+分组: {grline}')
print('DONE')
