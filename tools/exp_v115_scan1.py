# -*- coding: utf-8 -*-
"""实验1: 推理层参数扫描 — 校准/双指eff/密度阈值
评估: 官谱不动; 上架589偏差 + Spearman趋势
"""
import os, sys, io, pickle, numpy as np, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT

with open(os.path.join(_ROOT, 'models', '6dim_model_v11_4.pkl'), 'rb') as f:
    m = pickle.load(f)
gb, scaler = m['gb'], m['scaler']
FN, P95, P99 = m['feature_names'], m['p95_vals'], m['p99_vals']
LV_ORDER = m.get('lv_order', ['EZ','HD','IN','AT'])
CAPS = m.get('caps', {})
FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT)
_ALIGN = json.load(open(os.path.join(_ROOT, 'data', 'domain_align.json'), encoding='utf-8')).get('delta', {})
MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
DENS_FEATS = {'above_avg_density_mean', 'real_core_notes_per_second'}

def predict(feats_raw, level='IN', calib=None, eff_df=1.5, dens_th=9.5):
    feats = dict(feats_raw)
    lv = level.upper()
    if 'AT' in lv: lv = 'AT'
    elif 'IN' in lv: lv = 'IN'
    elif 'HD' in lv: lv = 'HD'
    else: lv = 'IN'
    if lv == 'IN':
        for k, d in _ALIGN.items():
            if k in feats: feats[k] = feats[k] - d
    if 'IN_AT' in LV_ORDER and lv in ('IN','AT'): lv = 'IN_AT'
    if lv not in LV_ORDER: lv = LV_ORDER[-1]
    vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(lv)] = 1.0
    x = np.array([[feats.get(n,0) for n in FN] + vec])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    mf3 = feats_raw.get('multi_finger_3plus_events', 0)
    dens = feats_raw.get('above_avg_density_mean', 0)
    ml = feats_raw.get('multi_line_sim_events', 0)
    wmf = feats_raw.get('weighted_mf_score_per_sec', 0)
    if mf3 >= 30 and ml >= 100:
        mf_scale, dens_s = 0.45, 0.85
    elif mf3 >= 30:
        mf_scale = (0.70 if dens >= dens_th else 0.50)
        dens_s = 1.0
    else:
        mf_scale, dens_s = (1.0 if mf3 <= 5 else 0.8), 1.0
    if mf3 <= 5:
        _sw = min(max((wmf - 12.0) / 6.0, 0.0), 1.0)
        if dens >= 10.0:
            eff_scale = 1.0
        else:
            eff_scale = eff_df - (eff_df - 1.0) * _sw
        wmf_scale = 1.0 - 0.4 * _sw
    else:
        eff_scale, wmf_scale = 1.0, 1.0
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
        co2 = co
        if fname in MF_FEATS: co2 = co2 * mf_scale
        if fname in EFF_FEATS: co2 = co2 * eff_scale
        if fname in DENS_FEATS and mf3 >= 30 and ml >= 100: co2 = co2 * dens_s
        if fname == 'weighted_mf_score_per_sec': co2 = co2 * wmf_scale
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
valid = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]

CONFIGS = [
    ('现状(校准0.40/0.25/0.05, eff1.5)', [(14,15,0.40),(15,16,0.25),(16,17,0.05)], 1.5),
    ('无校准', None, 1.5),
    ('无校准+eff1.7', None, 1.7),
    ('无校准+eff1.3', None, 1.3),
    ('轻校准(0.25/0.15/0.05)', [(14,15,0.25),(15,16,0.15),(16,17,0.05)], 1.5),
    ('无校准+dens_th8.5', None, 1.5),
]
for vname, calib, eff_df in CONFIGS:
    seg = {}
    for r in valid:
        d = r['diff']
        p = predict(r['feats'], r['level'], calib, eff_df)
        k = d < 14 and '<14' or d < 15 and '14-15' or d < 16 and '15-16' or d < 17 and '16-17' or '>=17'
        b = seg.setdefault(k, {'n':0,'b':0,'mae':0})
        b['n'] += 1; b['b'] += p-d; b['mae'] += abs(p-d)
    line = ' | '.join(f'{k}:{seg[k]["b"]/seg[k]["n"]:+.2f}' for k in ['<14','14-15','15-16','16-17','>=17'] if k in seg)
    # Spearman 趋势 (>=11 全部)
    ds = np.array([r['diff'] for r in valid]); ps = np.array([predict(r['feats'], r['level'], calib, eff_df) for r in valid])
    from scipy.stats import spearmanr
    rho, _ = spearmanr(ds, ps)
    print(f'{vname:<24} {line} | Spearman={rho:.3f}')
