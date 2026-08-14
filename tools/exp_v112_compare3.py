# -*- coding: utf-8 -*-
"""v11.0 vs v11.1 vs v11.2 统一对比
同一特征提取器(当前=v11.2状态) + 同一推理逻辑(app.py规则) + 各自pkl(GB/FLAT/P95)
"""
import os, sys, pickle, numpy as np, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT

MODELS = {
    'v11.0': '6dim_model_v11.pkl',
    'v11.1': '6dim_model_v11_1.pkl',
    'v11.2': '6dim_model_v11_2.pkl',
}
loaded = {}
for tag, fn in MODELS.items():
    with open(os.path.join(_ROOT, 'models', fn), 'rb') as f:
        loaded[tag] = pickle.load(f)

# 推理规则 (与 app.py v11.2 一致)
MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
DENS_FEATS = {'above_avg_density_mean', 'real_core_notes_per_second'}

def predict(m, feats_raw, level='IN'):
    gb, scaler = m['gb'], m['scaler']
    FN, P95, P99 = m['feature_names'], m['p95_vals'], m['p99_vals']
    LV_ORDER = m.get('lv_order', ['EZ','HD','IN','AT'])
    CAPS = m.get('caps', {})
    FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT)
    lv = level.upper()
    if 'IN_AT' in LV_ORDER and lv in ('IN','AT'): lv = 'IN_AT'
    if lv not in LV_ORDER: lv = LV_ORDER[-1]
    vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(lv)] = 1.0
    x = np.array([[feats_raw.get(n,0) for n in FN] + vec])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    mf3 = feats_raw.get('multi_finger_3plus_events', 0)
    dens = feats_raw.get('above_avg_density_mean', 0)
    ml = feats_raw.get('multi_line_sim_events', 0)
    wmf = feats_raw.get('weighted_mf_score_per_sec', 0)
    if mf3 >= 30 and ml >= 100:
        mf_scale, dens_s = 0.45, 0.85
    elif mf3 >= 30:
        mf_scale = 0.70 if dens >= 9.5 else 0.50
        dens_s = 1.0
    else:
        mf_scale = 1.0 if mf3 <= 5 else 0.8
        dens_s = 1.0
    df_stack = (mf3 <= 5 and wmf >= 15.0)
    eff_scale = 1.0 if mf3 >= 30 else (1.0 if df_stack else (1.5 if mf3 <= 5 else 1.0))
    wmf_scale = 0.6 if df_stack else 1.0
    total = 0.0
    cd_ = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats_raw.get(fname, 0)
        pv = P95.get(fname, 0)
        t = max(pv*0.55, bl*0.5)
        if v <= t: continue
        e = v/t - 1.0
        c = CAPS.get(fname, cd_)
        if c is not None and e > c: e = c
        co2 = co
        if fname in MF_FEATS: co2 = co * mf_scale
        elif fname in EFF_FEATS: co2 = co * eff_scale
        if fname in DENS_FEATS and mf3 >= 30 and ml >= 100: co2 = co * dens_s
        if fname == 'weighted_mf_score_per_sec': co2 = co * wmf_scale
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
        r4 = feats_raw.get('tracks_4plus_sec', 0) / act
        r5 = feats_raw.get('tracks_5plus_sec', 0) / act
        r6 = feats_raw.get('tracks_6plus_sec', 0) / act
        pred += 0.15 * min(r4, 0.8) + 0.55 * min(r5, 0.4) + 1.0 * min(r6, 0.15)
    for lo, hi, adj in [(14,15,0.30),(15,16,0.18),(16,17,0.05)]:
        if lo < pred <= hi: pred -= adj; break
    return pred

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)

# ===== 1. 官谱 in-sample (当前特征) =====
print('=== 官谱 in-sample (当前特征提取器) ===')
for tag, m in loaded.items():
    errs = []
    for f in cache['official']:
        lv = 'IN_AT' if f['level'] in ('IN','AT') else f['level']
        p = predict(m, f['feats'], lv)
        errs.append(p - f['diff'])
    errs = np.array(errs)
    print(f'  {tag}: MAE={np.abs(errs).mean():.4f} bias={errs.mean():+.4f}')

# ===== 2. 上架谱 589 =====
print('\n=== 上架谱 589 (pred - 社区diff) ===')
rkd = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
for tag, m in loaded.items():
    bins = {}
    for r in rkd:
        d = r['diff']
        p = predict(m, r['feats'], r['level'])
        bin_ = d < 13 and '<13' or d < 14 and '13-14' or d < 15 and '14-15' or d < 16 and '15-16' or d < 17 and '16-17' or '>=17'
        b = bins.setdefault(bin_, {'n':0,'b':0,'mae':0})
        b['n'] += 1; b['b'] += p-d; b['mae'] += abs(p-d)
    line = ' | '.join(f'{k}:{b["b"]/b["n"]:+.2f}/{b["mae"]/b["n"]:.2f}' for k, b in sorted(bins.items(), key=lambda x: float(x[0].replace('<','0').replace('-','.').replace('>=','99'))))
    print(f'  {tag}: {line}')
print('\nDONE')
