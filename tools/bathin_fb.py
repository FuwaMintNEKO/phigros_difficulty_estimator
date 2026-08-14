# -*- coding: utf-8 -*-
"""Bathin/Feeling Blue 详细特征诊断 (尾杀/长条差速/多面/读谱)"""
import os, sys, io, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])

# 找谱
targets = {}
for r in ranked:
    if r['id'] == 7516: targets['Bathin'] = r
    elif r['id'] == 47264: targets['Feeling Blue'] = r

# 特征: note_speed系列 + hold + ml + jline + 尾杀相关
KEYS = ['note_speed_non1_ratio','note_speed_std','note_speed_max','note_speed_density','speed_volatility','tempo_change_log_density','tempo_change_count',
        'multi_line_sim_events','hold_count','total_notes','hold_ratio','flash_hold_ratio','hold_interference_index','hold_lock_weighted_per_hold',
        'jline_movement_density','jline_rotate_density','jline_disappear_density','jline_relative_cross',
        'above_avg_duration_sec','eff_peak_tps_1s','above_avg_density_mean','fast_ms_050_ratio','fast_ms_100_ratio','fast_ms_150_ratio',
        'drag_per_sec','type_switch_per_sec','pattern_switch_rate','density_transition_std','density_transition_mean']
print(f'{"特征":<34}{"Bathin":>10}{"FeelBlue":>10}')
for k in KEYS:
    print(f'{k:<34}{targets["Bathin"]["feats"].get(k,0):>10.2f}{targets["Feeling Blue"]["feats"].get(k,0):>10.2f}')
print()
# P95 参考 (note_speed 特征触发情况)
print('P95 参考:')
for k in KEYS:
    if k in app_mod.P95:
        p95 = app_mod.P95[k]
        for nm, r in targets.items():
            v = r['feats'].get(k, 0)
            if v > p95 * 0.55:
                print(f'  {nm} {k}: v={v:.2f} P95={p95:.2f} 触发')
print()
# 顶贡献
for nm, r in targets.items():
    b, dims, kf = app_mod.compute_boost(dict(r['feats']), 1.0, is_custom=True)
    print(f'=== {nm} boost={b:.3f} ===')
    print('  cats:', {k: round(float(v),3) for k,v in dims['categories'].items()})
    print('  顶:', [(c[0], round(c[1],3)) for c in kf[:10]])
print('DONE')