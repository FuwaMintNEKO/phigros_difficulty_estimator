# -*- coding: utf-8 -*-
"""高仿 vs 官谱 特征逐项对比: 找预测差异来源"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

def feats_of(path, align_in=True):
    with open(path, 'rb') as f:
        cd, raw = load_chart_from_bytes(f.read())
    feats = extract_features(cd, speed=1.0)
    return feats

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
pairs = [
    ('夢の降る日に(高仿)', os.path.join(DL, '夢の降る日に', '5333883479687925.json'),
     '夢の降る日に(官谱IN)', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json'), 16.6),
    ('Der Schneid(高仿)', os.path.join(DL, 'Der Schneid(1)', '1903581575578621.json'),
     'DerSchneid(官谱AT)', os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json'), 17.5),
]
KEYS = ['multi_finger_3plus_events','weighted_mf_score_per_sec','real_notes_per_second','real_core_notes_per_second',
        'above_avg_density_mean','eff_peak_tps_1s','eff_avg_tps_1s','above_avg_duration_sec','hold_count','total_notes',
        'hold_ratio','drag_per_sec','type_switch_per_sec','chord_alternation_rate','jline_movement_density',
        'jline_rotate_density','movement_per_second','fast_ms_100_ratio','fast_ms_050_ratio','chord_jack_3plus_pairs',
        'stair_speed_avg','pattern_switch_rate','bpm','speed_volatility','multi_line_sim_events','hold_lock_weighted',
        'global_jack_count','jack_max_run']
for name1, p1, name2, p2, truth in pairs:
    f1 = feats_of(p1); f2 = feats_of(p2)
    print(f'\n=== {name1} vs {name2} (官谱定数={truth}) ===')
    print(f'{"特征":<34}{"高仿":>10}{"官谱":>10}')
    for k in KEYS:
        print(f'{k:<34}{f1.get(k,0):>10.2f}{f2.get(k,0):>10.2f}')
print('DONE')