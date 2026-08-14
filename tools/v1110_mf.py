# -*- coding: utf-8 -*-
"""v11.10 vs v11.13 对比: 双指/多指预测差异 (同一官谱集合)"""
import os, sys, io, json, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)

# 加载 v11.10 模型 (旧Flask用的)
with open(os.path.join(_ROOT, 'models', '6dim_model_v11_10.pkl'), 'rb') as f:
    m10 = pickle.load(f)
gb10, scl10, FN10 = m10['gb'], m10['scaler'], m10['feature_names']
LV10 = m10.get('lv_order', ['EZ','HD','IN','AT'])
FLAT10 = m10['MANUAL_FLAT']; P9510 = m10['p95_vals']; P9910 = m10['p99_vals']; CAPS10 = m10.get('caps', {})
CAL10 = [(14,15,0.51),(15,16,0.36),(16,17,0.16)]

import app as app_mod
charts = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
up_ids = {c['id'] for c in charts['上架']}
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10 and r['id'] in up_ids]
ds = np.array([round(r['diff'],1) for r in ranked])
int_mask = np.abs(ds - np.round(ds)) < 1e-6
ranked_f = [r for i, r in enumerate(ranked) if not int_mask[i]]
ds_f = ds[~int_mask]

# v11.10 预测 (完整管线: 含全段降权0.92 + 校准)
def predict_v1110(feats_raw, level_str):
    feats = dict(feats_raw)
    lv = level_str.upper()
    if 'AT' in lv: lv = 'AT'
    elif 'IN' in lv: lv = 'IN'
    elif 'HD' in lv: lv = 'HD'
    else: lv = 'IN'
    if lv == 'IN':
        for k, d in app_mod.DOMAIN_DELTA.items():
            if k in feats: feats[k] = feats[k] - d
    lv2 = 'IN_AT' if lv in ('IN','AT') and 'IN_AT' in LV10 else lv
    if lv2 not in LV10: lv2 = LV10[-1]
    vec = [0.0]*len(LV10); vec[LV10.index(lv2)] = 1.0
    x = np.array([[feats.get(n,0) for n in FN10] + vec])
    p_gb = float(gb10.predict(scl10.transform(x))[0])
    MF = {'weighted_mf_score_per_sec','multi_finger_3plus_events','discrete_mf_ratio','chord_alternation_rate'}
    EFF = {'eff_peak_tps_1s','eff_avg_tps_1s'}
    DENS = {'above_avg_density_mean','real_core_notes_per_second'}
    EXT = {'cross_hand_density','jline_relative_cross','thirtysecond_run_max','thirtysecond_run_ratio','lane_switch_density'}
    mf3 = feats_raw.get('multi_finger_3plus_events', 0); dens = feats_raw.get('above_avg_density_mean', 0)
    ml = feats_raw.get('multi_line_sim_events', 0); wmf = feats_raw.get('weighted_mf_score_per_sec', 0)
    if mf3 >= 30 and ml >= 100: mf_scale, dens_s = 0.45, 0.85
    elif mf3 >= 30: mf_scale, dens_s = (0.70 if dens >= 9.5 else 0.50), 1.0
    else: mf_scale, dens_s = (1.0 if mf3 <= 5 else 0.8), 1.0
    if mf3 <= 5:
        _sw = min(max((wmf - 12.0)/6.0, 0.0), 1.0)
        eff_scale = 1.0 if dens >= 10.0 else 1.5 - 0.5*_sw
        wmf_scale = 1.0 - 0.4*_sw
        extreme_scale = 1.3
    elif mf3 >= 30:
        eff_scale, wmf_scale, extreme_scale = 1.0, 1.0, 0.7
    else:
        eff_scale, wmf_scale, extreme_scale = 1.0, 1.0, 1.0
    ts = app_mod.compute_tags(feats)
    high = {'叠键','多押','变速','位移'}
    stack = 0.92 if sum(1 for t in ts if t in high) >= 2 else 1.0
    total = 0.0; cd_ = CAPS10.get('_default', None)
    for fname, bl, co in FLAT10:
        v = feats.get(fname, 0); pv = P9510.get(fname, 0)
        t = max(pv*0.55, bl*0.5)
        if v <= t: continue
        e = v/t - 1.0
        c = CAPS10.get(fname, cd_)
        if c is not None and e > c: e = c
        co2 = co
        if fname in MF: co2 = co * mf_scale
        elif fname in EFF: co2 = co * eff_scale
        if fname in DENS and mf3 >= 30 and ml >= 100: co2 = co * dens_s
        if fname == 'weighted_mf_score_per_sec': co2 = co * wmf_scale
        if fname in EXT: co2 = co * extreme_scale
        co2 = co2 * stack
        x_ = co2 * (e**0.70)
        p99 = max(P9910.get(fname,0), bl*0.5)
        if v > p99:
            pe = v/p99 - 1.0
            if c is not None and pe > c: pe = c
            x_ += co2*max(0,pe)**0.70*0.5
        total += x_
    pred = p_gb + total
    act = feats_raw.get('tracks_active_sec', 0)
    if act > 0:
        pred += 0.15*min(feats_raw.get('tracks_4plus_sec',0)/act,0.8) + 0.55*min(feats_raw.get('tracks_5plus_sec',0)/act,0.4) + 1.0*min(feats_raw.get('tracks_6plus_sec',0)/act,0.15)
    hr = feats_raw.get('hold_count', 0)/max(feats_raw.get('total_notes',1),1)
    if hr >= 0.6: pred += 0.7
    elif hr >= 0.4: pred += 0.5
    elif hr >= 0.25: pred += 0.3
    for lo, hi, adj in CAL10:
        if lo < pred <= hi: pred -= adj; break
    return pred

ps10 = np.array([predict_v1110(r['feats'], r['level']) for r in ranked_f])
errs10 = ps10 - ds_f
mf3 = np.array([r['feats'].get('multi_finger_3plus_events', 0) for r in ranked_f])
print('=== v11.10 (旧Flask) 按 mf3 分组 ===')
for lo, hi, tag in [(0,5,'双指'), (6,29,'混合'), (30,99,'多指')]:
    mk = np.where((mf3 >= lo) & (mf3 < hi))[0]
    print(f'  {tag:<8} n={len(mk):>3} bias={errs10[mk].mean():+.3f} MAE={np.abs(errs10[mk]).mean():.3f}')
print('\n=== v11.10 高难段(>=16.5) 按 mf3 ===')
for lo, hi, tag in [(0,5,'双指'), (6,29,'混合'), (30,99,'多指')]:
    mk = np.where((mf3 >= lo) & (mf3 < hi) & (ds_f >= 16.5))[0]
    if len(mk):
        print(f'  {tag:<8} n={len(mk):>3} bias={errs10[mk].mean():+.3f} MAE={np.abs(errs10[mk]).mean():.3f}')
print('DONE')