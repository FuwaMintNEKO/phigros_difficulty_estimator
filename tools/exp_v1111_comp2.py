# -*- coding: utf-8 -*-
"""逐特征对比 The wheel to the right: app vs manual"""
import os, sys, numpy as np, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
r = next(x for x in ranked if x['name'] == 'The wheel to the right')
feats = dict(r['feats'])
FLAT = app_mod.MANUAL_FLAT; P95 = app_mod.P95; P99 = app_mod.P99; CAPS = app_mod.CAPS
MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
DENS_FEATS = {'above_avg_density_mean', 'real_core_notes_per_second'}
EXTREME_FEATS = {'cross_hand_density', 'jline_relative_cross', 'thirtysecond_run_max', 'thirtysecond_run_ratio', 'lane_switch_density'}
mf3 = feats.get('multi_finger_3plus_events', 0); dens = feats.get('above_avg_density_mean', 0)
ml = feats.get('multi_line_sim_events', 0); wmf = feats.get('weighted_mf_score_per_sec', 0)
print(f'mf3={mf3} ml={ml} dens={dens:.2f} wmf={wmf:.2f}')
# app 参数
if mf3 >= 30 and ml >= app_mod.ML_HEAVY_TH: mf_scale, dens_s = app_mod.ML_HEAVY_MF, app_mod.ML_HEAVY_DENS
elif mf3 >= 30: mf_scale, dens_s = (app_mod.MF3_SCALE_HIDENS if dens >= app_mod.MF3_HIDENS_TH else app_mod.MF3_SCALE_GE30), 1.0
else: mf_scale, dens_s = (1.0 if mf3 <= 5 else app_mod.MF3_SCALE_MID), 1.0
if mf3 <= 5:
    _sw = min(max((wmf - app_mod.DF_STACK_WMF_LO) / (app_mod.DF_STACK_WMF_HI - app_mod.DF_STACK_WMF_LO), 0.0), 1.0)
    eff_scale = app_mod.EFF_SCALE_DF_STACK if dens >= 10.0 else app_mod.EFF_SCALE_LE5 - (app_mod.EFF_SCALE_LE5 - app_mod.EFF_SCALE_DF_STACK) * _sw
    wmf_scale = 1.0 - (1.0 - app_mod.DF_WMF_SCALE) * _sw
    extreme_scale = app_mod.EXTREME_SCALE_DF
elif mf3 >= 30:
    eff_scale, wmf_scale, extreme_scale = 1.0, 1.0, app_mod.EXTREME_SCALE_MF
else:
    eff_scale, wmf_scale, extreme_scale = 1.0, 1.0, 1.0
print(f'mf_scale={mf_scale} dens_s={dens_s} eff={eff_scale} wmf_scale={wmf_scale} ext={extreme_scale}')

def contrib(fname, bl, co):
    v = feats.get(fname, 0); pv = P95.get(fname, 0)
    t = max(pv * 0.55, bl * 0.5)
    if v <= t: return 0.0
    e = v / t - 1.0
    c = CAPS.get(fname, CAPS.get('_default', None))
    if c is not None and e > c: e = c
    co2 = co
    if fname in MF_FEATS: co2 = co * mf_scale
    elif fname in EFF_FEATS: co2 = co * eff_scale
    if fname in DENS_FEATS and mf3 >= 30 and ml >= app_mod.ML_HEAVY_TH: co2 = co * dens_s
    if fname == 'weighted_mf_score_per_sec': co2 = co * wmf_scale
    if fname in EXTREME_FEATS: co2 = co * extreme_scale
    x = co2 * (e ** 0.70)
    p99 = max(P99.get(fname, 0), bl * 0.5)
    if v > p99:
        pe = v / p99 - 1.0
        if c is not None and pe > c: pe = c
        x += co2 * max(0, pe) ** 0.70 * 0.5
    return x, co2, e

print(f'\n{"特征":<32}{"app贡献":>10}{"手动贡献":>10}')
b_app, _, contribs = app_mod.compute_boost(feats, 1.0, is_custom=True)
cmap = {c[0]: c[1] for c in contribs}
total_man = 0.0
for fname, bl, co in FLAT:
    x2, co2, e = contrib(fname, bl, co)
    if x2 > 0:
        total_man += x2
        diff = cmap.get(fname, 0) - x2
        if abs(diff) > 0.001:
            print(f'{fname:<32}{cmap.get(fname,0):>10.4f}{x2:>10.4f}  co2={co2:.4f}')
print(f'\napp_total={b_app:.4f} manual_total={total_man:.4f}')
print('DONE')