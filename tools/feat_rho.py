# -*- coding: utf-8 -*-
"""特征计算审查: 官谱中 各特征与定数的相关性 + 是否有异常值"""
import os, sys, io, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from scipy.stats import spearmanr
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = cache['official']
# 15+ 官谱
off15 = [o for o in official if o['diff'] >= 15]
y = np.array([o['diff'] for o in off15])
print(f'官谱15+ : {len(off15)}')

# 关键特征与定数相关性
KEYS = ['micro_max_0.0625beat','micro_max_0.125beat','miniburst_count','miniburst_density','global_jack_count',
        'fast_ms_050_ratio','fast_ms_100_ratio','fast_ms_150_ratio','real_notes_per_second','real_core_notes_per_second',
        'above_avg_density_mean','eff_peak_tps_1s','eff_avg_tps_1s','above_avg_duration_sec','movement_per_second',
        'movement_density_index','cross_hand_density','lane_switch_density','speed_volatility','type_switch_per_sec',
        'chord_alternation_rate','rhythm_entropy','stair_speed_avg','pattern_switch_rate','hold_lock_weighted',
        'multi_line_sim_events','jline_movement_density','jline_rotate_density','density_transition_std']
print(f'{"特征":<34}{"rho(15+官谱)":>12}{"P50":>8}{"P90":>8}{"max":>10}')
for k in KEYS:
    vals = np.array([o['feats'].get(k, 0) for o in off15])
    if vals.std() == 0:
        print(f'{k:<34}{"恒定":>12}{np.median(vals):>8.1f}{np.percentile(vals,90):>8.1f}{vals.max():>10.1f}')
        continue
    rho = spearmanr(vals, y).statistic
    print(f'{k:<34}{rho:>12.3f}{np.median(vals):>8.1f}{np.percentile(vals,90):>8.1f}{vals.max():>10.1f}')
print('DONE')