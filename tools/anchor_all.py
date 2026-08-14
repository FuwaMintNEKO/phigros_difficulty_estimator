# -*- coding: utf-8 -*-
"""5锚点完整数据: 原始特征 + 贡献"""
import os, sys, io, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

def feats_of(path, lv_str):
    with open(path, 'rb') as f:
        cd, raw = load_chart_from_bytes(f.read())
    feats = extract_features(cd, speed=1.0)
    lv = 'AT' if 'AT' in lv_str.upper() else ('IN' if 'IN' in lv_str.upper() else 'HD')
    if lv == 'IN':
        for k, d in app_mod.DOMAIN_DELTA.items():
            if k in feats: feats[k] = feats[k] - d
    return feats

# 各锚点: (名称, 锚点, 当前pred, feats来源, level)
anchors = []
# ranked 三个
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
for r in cache['ranked']:
    if r['id'] == 7516:
        anchors.append(('Bathin', 17.2, r['feats'], r['level']))
    elif r['id'] == 59064:
        anchors.append(('ずんどこ', 15.8, r['feats'], r['level']))
    elif r['id'] == 15875:
        anchors.append(('FREEDOM DiVE', 16.15, r['feats'], r['level']))
anchors.append(('Apollo', 18.0, feats_of(os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '41242.json'), 'AT'), 'AT'))
anchors.append(('Chart_SP#1347', 17.65, feats_of(os.path.join(_ROOT, 'data', 'test_charts', 'Chart_SP #1347(1).json'), 'IN'), 'IN'))

KEYS = ['real_core_notes_per_second','real_notes_per_second','above_avg_density_mean','eff_peak_tps_1s','eff_avg_tps_1s',
        'weighted_mf_score_per_sec','type_switch_per_sec','drag_per_sec','drag_flick_ratio','above_avg_duration_sec',
        'chord_alternation_rate','jline_movement_density','stair_speed_avg','pattern_switch_rate','multi_finger_3plus_events',
        'hold_count','total_notes','movement_per_second','rhythm_entropy','jack_max_run','density_transition_std',
        'cross_hand_density','lane_switch_density','global_jack_count','fast_ms_100_ratio','fast_ms_050_ratio']
print('=== 原始特征 ===')
print(f'{"特征":<32}' + ''.join(f'{n[:6]:>9}' for n, _, _, _ in anchors))
for k in KEYS:
    row = f'{k:<32}'
    for nm, tgt, feats, lv in anchors:
        row += f'{feats.get(k,0):>9.2f}'
    print(row)
print()
print('=== 预测 vs 锚点 ===')
print(f'{"谱":<16}{"锚点":>7}{"当前":>7}{"差":>7}')
for nm, tgt, feats, lv in anchors:
    # 用完整管线
    pred, _, _, _, _ = full_predict(feats, lv)
    print(f'{nm:<16}{tgt:>7.2f}{pred:>7.2f}{pred-tgt:>+7.2f}')
print('DONE')