# -*- coding: utf-8 -*-
"""Melodiniq vs 同段双指谱: 特征对比, 找抬升方向"""
import os, sys, io, json, csv, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

# Melodiniq 完整特征
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, raw = load_chart_from_bytes(f.read())
mel = extract_features(cd, speed=1.0)
print('=== Melodiniq (#61184) 特征 ===')
KEYS = ['multi_finger_3plus_events','weighted_mf_score_per_sec','real_notes_per_second','real_core_notes_per_second',
        'above_avg_density_mean','eff_peak_tps_1s','eff_avg_tps_1s','above_avg_duration_sec','hold_count','total_notes',
        'hold_ratio','drag_per_sec','type_switch_per_sec','chord_alternation_rate','jline_movement_density',
        'jline_rotate_density','jline_disappear_density','jline_relative_cross','movement_per_second','movement_density_index',
        'fast_ms_100_ratio','fast_ms_050_ratio','chord_jack_3plus_pairs','stair_speed_avg','pattern_switch_rate',
        'bpm','speed_volatility','multi_line_sim_events','hold_lock_weighted','global_jack_count','jack_max_run',
        'long_jack_count','same_line_jack_ratio','cross_hand_density','lane_switch_density','rhythm_entropy',
        'note_clutter_ratio','density_transition_std','density_transition_mean','flash_hold_ratio']
for k in KEYS:
    v = mel.get(k, 0)
    p95 = app_mod.P95.get(k, 0)
    thr = max(p95*0.55, 0)
    flag = ' <== 触发' if v > thr and thr > 0 else ''
    print(f'  {k:<34} v={v:>10.2f} P95={p95:>9.2f} 阈值={thr:>8.2f}{flag}')
print('DONE')