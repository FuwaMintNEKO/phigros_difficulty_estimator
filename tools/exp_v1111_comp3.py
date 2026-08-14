# -*- coding: utf-8 -*-
"""精确对比: 直接复制app循环逻辑, 逐特征打app/manual缩放系数差异"""
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

# 手动: 与 app 完全相同的结构 (is_custom=True)
def manual_loop():
    total = 0.0; cap_default = CAPS.get('_default', None)
    mf3 = feats.get('multi_finger_3plus_events', 0); dens = feats.get('above_avg_density_mean', 0)
    ml = feats.get('multi_line_sim_events', 0)
    if mf3 >= 30 and ml >= app_mod.ML_HEAVY_TH:
        mf_scale = app_mod.ML_HEAVY_MF; dens_scale_ml = app_mod.ML_HEAVY_DENS
    elif mf3 >= 30:
        mf_scale = app_mod.MF3_SCALE_HIDENS if dens >= app_mod.MF3_HIDENS_TH else app_mod.MF3_SCALE_GE30
        dens_scale_ml = 1.0
    else:
        mf_scale = 1.0 if mf3 <= 5 else app_mod.MF3_SCALE_MID
    if mf3 <= 5:
        _w = feats.get('weighted_mf_score_per_sec', 0)
        _sw = min(max((_w - app_mod.DF_STACK_WMF_LO) / (app_mod.DF_STACK_WMF_HI - app_mod.DF_STACK_WMF_LO), 0.0), 1.0)
        if dens >= 10.0: eff_scale = app_mod.EFF_SCALE_DF_STACK
        else: eff_scale = app_mod.EFF_SCALE_LE5 - (app_mod.EFF_SCALE_LE5 - app_mod.EFF_SCALE_DF_STACK) * _sw
        wmf_scale = 1.0 - (1.0 - app_mod.DF_WMF_SCALE) * _sw
        extreme_scale = app_mod.EXTREME_SCALE_DF
    elif mf3 >= 30:
        eff_scale, wmf_scale, extreme_scale = 1.0, 1.0, app_mod.EXTREME_SCALE_MF
    else:
        eff_scale, wmf_scale, extreme_scale = 1.0, 1.0, 1.0
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0); pv = P95.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t: continue
        e = v / t - 1.0
        c = CAPS.get(fname, cap_default)
        if c is not None and e > c: e = c
        co0 = co
        if fname in app_mod.MF_FEATS_COND: co = co0 * mf_scale
        elif fname in app_mod.EFF_FEATS_COND: co = co0 * eff_scale
        if fname in app_mod.DENS_FEATS_COND and mf3 >= 30 and ml >= app_mod.ML_HEAVY_TH: co = co0 * dens_scale_ml
        if fname == 'weighted_mf_score_per_sec': co = co0 * wmf_scale
        if fname in app_mod.EXTREME_FEATS_COND: co = co0 * extreme_scale
        x = co * (e ** 0.70)
        p99 = max(P99.get(fname, 0), bl * 0.5)
        if v > p99:
            pe = v / p99 - 1.0
            if c is not None and pe > c: pe = c
            x += co * max(0, pe) ** 0.70 * 0.5
        total += x
    return total

b_app, _, _ = app_mod.compute_boost(feats, 1.0, is_custom=True)
b_man = manual_loop()
print(f'app={b_app:.6f} manual={b_man:.6f} diff={b_app-b_man:+.6f}')

# 再验证: 直接调用 app 内部能否拿到逐特征? 打印 app 的contribs全部(不截断)
# 修改: 手动模拟 app 的每个特征, 对比 app 的 contribs
import types
# 重新实现: 打印每个特征 app贡献 vs manual贡献
def manual_contribs():
    out = {}; cap_default = CAPS.get('_default', None)
    mf3 = feats.get('multi_finger_3plus_events', 0); dens = feats.get('above_avg_density_mean', 0)
    ml = feats.get('multi_line_sim_events', 0)
    if mf3 >= 30 and ml >= app_mod.ML_HEAVY_TH:
        mf_scale = app_mod.ML_HEAVY_MF; dens_scale_ml = app_mod.ML_HEAVY_DENS
    elif mf3 >= 30:
        mf_scale = app_mod.MF3_SCALE_HIDENS if dens >= app_mod.MF3_HIDENS_TH else app_mod.MF3_SCALE_GE30
        dens_scale_ml = 1.0
    else:
        mf_scale = 1.0 if mf3 <= 5 else app_mod.MF3_SCALE_MID
    if mf3 <= 5:
        _w = feats.get('weighted_mf_score_per_sec', 0)
        _sw = min(max((_w - app_mod.DF_STACK_WMF_LO) / (app_mod.DF_STACK_WMF_HI - app_mod.DF_STACK_WMF_LO), 0.0), 1.0)
        if dens >= 10.0: eff_scale = app_mod.EFF_SCALE_DF_STACK
        else: eff_scale = app_mod.EFF_SCALE_LE5 - (app_mod.EFF_SCALE_LE5 - app_mod.EFF_SCALE_DF_STACK) * _sw
        wmf_scale = 1.0 - (1.0 - app_mod.DF_WMF_SCALE) * _sw
        extreme_scale = app_mod.EXTREME_SCALE_DF
    elif mf3 >= 30:
        eff_scale, wmf_scale, extreme_scale = 1.0, 1.0, app_mod.EXTREME_SCALE_MF
    else:
        eff_scale, wmf_scale, extreme_scale = 1.0, 1.0, 1.0
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0); pv = P95.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t: continue
        e = v / t - 1.0
        c = CAPS.get(fname, cap_default)
        if c is not None and e > c: e = c
        co0 = co
        if fname in app_mod.MF_FEATS_COND: co = co0 * mf_scale
        elif fname in app_mod.EFF_FEATS_COND: co = co0 * eff_scale
        if fname in app_mod.DENS_FEATS_COND and mf3 >= 30 and ml >= app_mod.ML_HEAVY_TH: co = co0 * dens_scale_ml
        if fname == 'weighted_mf_score_per_sec': co = co0 * wmf_scale
        if fname in app_mod.EXTREME_FEATS_COND: co = co0 * extreme_scale
        x = co * (e ** 0.70)
        p99 = max(P99.get(fname, 0), bl * 0.5)
        if v > p99:
            pe = v / p99 - 1.0
            if c is not None and pe > c: pe = c
            x += co * max(0, pe) ** 0.70 * 0.5
        out[fname] = x
    return out

mc = manual_contribs()
# app contribs: 需要全部, 不截断
# 通过 monkey patch 抓取? 简化: 重算 app 逻辑但打印
print('\n特征差异 (app-manual):')
diffs = []
for fname, bl, co in FLAT:
    v = feats.get(fname, 0); pv = P95.get(fname, 0)
    t = max(pv * 0.55, bl * 0.5)
    if v > t:
        diffs.append((fname, mc.get(fname, 0)))
# 无法直接拿app每特征, 只能从 total 对比
print('DONE')