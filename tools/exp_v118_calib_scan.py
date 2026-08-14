# -*- coding: utf-8 -*-
"""校准表优化扫描: ranked bias 归零 (含17+段)
"""
import os, sys, pickle, numpy as np, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from scipy.stats import spearmanr
from boost_config import MANUAL_FLAT
m = pickle.load(open(os.path.join(_ROOT, 'models', '6dim_model_v11_7b.pkl'), 'rb'))
gb, scaler = m['gb'], m['scaler']; FN = m['feature_names']; LV_ORDER = m['lv_order']
P95 = m['p95_vals']; P99 = m['p99_vals']; FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT); CAPS = m.get('caps', {})
_ALIGN = json.load(open(os.path.join(_ROOT, 'data', 'domain_align.json'), encoding='utf-8')).get('delta', {})
MF_FEATS_COND = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS_COND = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
DENS_FEATS_COND = {'above_avg_density_mean', 'real_core_notes_per_second'}
EXTREME_FEATS_COND = {'cross_hand_density', 'jline_relative_cross', 'thirtysecond_run_max', 'thirtysecond_run_ratio', 'lane_switch_density'}
def predict(feats_raw, level='IN', calib=None):
    feats = dict(feats_raw)
    lv = level.upper()
    if 'AT' in lv: lv = 'AT'
    elif 'IN' in lv: lv = 'IN'
    elif 'HD' in lv: lv = 'HD'
    elif 'EZ' in lv: lv = 'EZ'
    else: lv = 'IN'
    if lv == 'IN':
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
        extreme_scale = 0.7
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
    if calib:
        for lo, hi, adj in calib:
            if lo < pred <= hi: pred -= adj; break
    return pred

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([r['diff'] for r in ranked])
CONFIGS = [
    ('现状(0.55/0.40/0.20)', [(14,15,0.55),(15,16,0.40),(16,17,0.20)]),
    ('A: 0.40/0.50/0.55+17:0.55', [(14,15,0.40),(15,16,0.50),(16,17,0.55),(17,99,0.55)]),
    ('B: 0.38/0.48/0.52+17:0.55', [(14,15,0.38),(15,16,0.48),(16,17,0.52),(17,99,0.55)]),
    ('C: 0.42/0.52/0.58+17:0.60', [(14,15,0.42),(15,16,0.52),(16,17,0.58),(17,99,0.60)]),
    ('D: 0.35/0.45/0.50+17:0.50', [(14,15,0.35),(15,16,0.45),(16,17,0.50),(17,99,0.50)]),
]
for tag, calib in CONFIGS:
    ps = np.array([predict(r['feats'], r['level'], calib) for r in ranked])
    errs = ps - ds
    rho, _ = spearmanr(ds, ps)
    seg = []
    for lo, hi, t2 in [(10,14,'<14'),(14,15,'14-15'),(15,16,'15-16'),(16,17,'16-17'),(17,99,'>=17')]:
        mk = np.where((ds >= lo) & (ds < hi))[0]
        if len(mk): seg.append(f'{t2}:{errs[mk].mean():+.2f}')
    print(f'{tag:<28} MAE={np.abs(errs).mean():.3f} bias={errs.mean():+.3f} rho={rho:.3f} | {" ".join(seg)}')
print('DONE')
