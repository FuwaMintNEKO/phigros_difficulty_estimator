# -*- coding: utf-8 -*-
"""逐特征: 复制app.compute_boost循环, 打印每特征贡献 vs 实际app"""
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

# monkey-patch: 抓 app.compute_boost 每特征贡献
orig_append = None
captured = []
import app as _m
# 直接复制源码逻辑并逐特征打印
FLAT = app_mod.MANUAL_FLAT; P95 = app_mod.P95; P99 = app_mod.P99; CAPS = app_mod.CAPS
cap_default = CAPS.get('_default', None)
excess_exp = 0.70
mf3 = feats.get('multi_finger_3plus_events', 0); dens = feats.get('above_avg_density_mean', 0)
ml = feats.get('multi_line_sim_events', 0)
print(f'mf3={mf3} ml={ml} dens={dens:.3f}')
if mf3 >= 30 and ml >= app_mod.ML_HEAVY_TH:
    mf_scale = app_mod.ML_HEAVY_MF; dens_scale_ml = app_mod.ML_HEAVY_DENS
    print('档: 多面型')
elif mf3 >= 30:
    mf_scale = app_mod.MF3_SCALE_HIDENS if dens >= app_mod.MF3_HIDENS_TH else app_mod.MF3_SCALE_GE30
    dens_scale_ml = 1.0
    print(f'档: 多指 mf_scale={mf_scale}')
else:
    mf_scale = 1.0 if mf3 <= 5 else app_mod.MF3_SCALE_MID
    print(f'档: 其他 mf_scale={mf_scale}')
if mf3 <= 5:
    _w = feats.get('weighted_mf_score_per_sec', 0)
    _sw = min(max((_w - app_mod.DF_STACK_WMF_LO) / (app_mod.DF_STACK_WMF_HI - app_mod.DF_STACK_WMF_LO), 0.0), 1.0)
    eff_scale = app_mod.EFF_SCALE_DF_STACK if dens >= 10.0 else app_mod.EFF_SCALE_LE5 - (app_mod.EFF_SCALE_LE5 - app_mod.EFF_SCALE_DF_STACK) * _sw
    wmf_scale = 1.0 - (1.0 - app_mod.DF_WMF_SCALE) * _sw
    extreme_scale = app_mod.EXTREME_SCALE_DF
elif mf3 >= 30:
    eff_scale, wmf_scale, extreme_scale = 1.0, 1.0, app_mod.EXTREME_SCALE_MF
else:
    eff_scale, wmf_scale, extreme_scale = 1.0, 1.0, 1.0
_stack_scale = 1.0
total = 0.0
rows = []
for fname, bl, co in FLAT:
    v = feats.get(fname, 0); pv = P95.get(fname, 0)
    t = max(pv * 0.55, bl * 0.5)
    if v <= t: continue
    e = v / t - 1.0
    c = CAPS.get(fname, cap_default)
    if c is not None and e > c: e = c
    co0 = co
    if fname in app_mod.MF_FEATS_COND: co = co * mf_scale
    elif fname in app_mod.EFF_FEATS_COND: co = co * eff_scale
    if fname in app_mod.DENS_FEATS_COND and mf3 >= 30 and ml >= app_mod.ML_HEAVY_TH: co = co * dens_scale_ml
    if fname == 'weighted_mf_score_per_sec': co = co * wmf_scale
    if fname in app_mod.EXTREME_FEATS_COND: co = co * extreme_scale
    co = co * _stack_scale
    x = co * (e ** excess_exp)
    p99 = max(P99.get(fname, 0), bl * 0.5)
    if v > p99:
        pe = v / p99 - 1.0
        if c is not None and pe > c: pe = c
        x += co * max(0, pe) ** excess_exp * 0.5
    total += x
    rows.append((fname, x, co0, co))
print(f'manual_total={total:.6f}  app_total=3.018228')
# 对比 app 的 contribs (top15 但有全部吗? compute_boost返回contribs[:15])
b_app, dims, c15 = app_mod.compute_boost(feats, 1.0, is_custom=True)
cmap = {c[0]: c[1] for c in c15}
print(f'\n{"特征":<34}{"manual":>10}{"app(top15)":>12}')
for fname, x, co0, co in sorted(rows, key=lambda z: -z[1]):
    print(f'{fname:<34}{x:>10.4f}{cmap.get(fname, 0):>12.4f}')
print('DONE')