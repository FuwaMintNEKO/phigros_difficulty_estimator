# -*- coding: utf-8 -*-
"""Chart_SP#1347 vs Apollo 完整特征贡献对比"""
import os, sys, io, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

def get_feats(path, lv_str):
    with open(path, 'rb') as f:
        cd, raw = load_chart_from_bytes(f.read())
    feats = extract_features(cd, speed=1.0)
    lv = 'AT' if 'AT' in lv_str.upper() else ('IN' if 'IN' in lv_str.upper() else 'HD')
    if lv == 'IN':
        for k, d in app_mod.DOMAIN_DELTA.items():
            if k in feats: feats[k] = feats[k] - d
    return feats

A = get_feats(os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '41242.json'), 'AT')
C = get_feats(os.path.join(_ROOT, 'data', 'test_charts', 'Chart_SP #1347(1).json'), 'IN')

bA, dA, kfA = app_mod.compute_boost(A, 1.0, is_custom=True)
bC, dC, kfC = app_mod.compute_boost(C, 1.0, is_custom=True)
print(f'Apollo boost={bA:.3f}  ChartSP boost={bC:.3f}')
# 合并所有特征名, 对比贡献
names = set(c[0] for c in kfA) | set(c[0] for c in kfC)
mA = {c[0]: c for c in kfA}; mC = {c[0]: c for c in kfC}
print(f'{"特征":<32}{"Apollo贡献":>10}{"ChartSP贡献":>12}{"差值":>8}')
for n in sorted(names, key=lambda n: -(mA.get(n,[0,0])[1] + mC.get(n,[0,0])[1])):
    va = mA.get(n, [n,0,0,0,0])[1]; vc = mC.get(n, [n,0,0,0,0])[1]
    print(f'{n:<32}{va:>10.3f}{vc:>12.3f}{vc-va:>+8.3f}')
print('\n关键原始特征:')
for k in ['real_core_notes_per_second','real_notes_per_second','above_avg_density_mean','eff_peak_tps_1s','weighted_mf_score_per_sec','type_switch_per_sec','drag_per_sec','above_avg_duration_sec','chord_alternation_rate','jline_movement_density','stair_speed_avg','pattern_switch_rate','multi_finger_3plus_events','hold_count','total_notes','movement_per_second','rhythm_entropy']:
    print(f'  {k:<34} A={A.get(k,0):>10.2f}  C={C.get(k,0):>10.2f}')
print('DONE')