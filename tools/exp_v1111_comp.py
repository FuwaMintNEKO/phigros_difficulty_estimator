# -*- coding: utf-8 -*-
"""逐特征对比 app.compute_boost vs 手动循环 (找差异)"""
import os, sys, numpy as np, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])

def manual_boost(feats_raw):
    feats = dict(feats_raw)
    FLAT = app_mod.MANUAL_FLAT; P95 = app_mod.P95; P99 = app_mod.P99; CAPS = app_mod.CAPS
    MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
    EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
    DENS_FEATS = {'above_avg_density_mean', 'real_core_notes_per_second'}
    EXTREME_FEATS = {'cross_hand_density', 'jline_relative_cross', 'thirtysecond_run_max', 'thirtysecond_run_ratio', 'lane_switch_density'}
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
        if fname in MF_FEATS: co2 = co * mf_scale
        elif fname in EFF_FEATS: co2 = co * eff_scale
        if fname in DENS_FEATS and mf3 >= 30 and ml >= 100: co2 = co * dens_s
        if fname == 'weighted_mf_score_per_sec': co2 = co * wmf_scale
        if fname in EXTREME_FEATS: co2 = co * extreme_scale
        x_ = co2 * (e**0.70)
        p99 = max(P99.get(fname,0), bl*0.5)
        if v > p99:
            pe = v/p99 - 1.0
            if c is not None and pe > c: pe = c
            x_ += co2*max(0,pe)**0.70*0.5
        total += x_
    return total

# 对比每首谱
maxdiff = 0; worst = None
n_diff = 0
for i, r in enumerate(ranked):
    b_app, _, _ = app_mod.compute_boost(dict(r['feats']), 1.0, is_custom=True)
    b_man = manual_boost(r['feats'])
    d = abs(b_app - b_man)
    if d > 1e-6:
        n_diff += 1
        if d > maxdiff:
            maxdiff = d; worst = (r['name'], ds[i], b_app, b_man)
print(f'差异谱数: {n_diff}/{len(ranked)}  maxdiff={maxdiff}')
if worst: print('worst:', worst)
# 找一首差异最大的, 逐特征对比
if worst:
    for r in ranked:
        if r['name'] == worst[0]:
            b_app, _, _ = app_mod.compute_boost(dict(r['feats']), 1.0, is_custom=True)
            b_man = manual_boost(r['feats'])
            print(f'\n{worst[0]}  app={b_app:.4f} manual={b_man:.4f}')
            # 手动重算每特征
            FLAT = app_mod.MANUAL_FLAT; P95 = app_mod.P95; P99 = app_mod.P99; CAPS = app_mod.CAPS
            MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
            EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
            DENS_FEATS = {'above_avg_density_mean', 'real_core_notes_per_second'}
            EXTREME_FEATS = {'cross_hand_density', 'jline_relative_cross', 'thirtysecond_run_max', 'thirtysecond_run_ratio', 'lane_switch_density'}
            mf3 = r['feats'].get('multi_finger_3plus_events', 0); dens = r['feats'].get('above_avg_density_mean', 0)
            ml = r['feats'].get('multi_line_sim_events', 0); wmf = r['feats'].get('weighted_mf_score_per_sec', 0)
            print(f'mf3={mf3} ml={ml} dens={dens} wmf={wmf}')
            break
print('DONE')