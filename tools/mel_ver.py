# -*- coding: utf-8 -*-
"""Melodiniq vs 官谱Verrückt(癫狂) 特征对比 + BPM/24分分析"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

# 找 Verrückt 官谱
import glob
charts = glob.glob(os.path.join(_ROOT, 'data', 'chart', '*'))
ver = None
for c in charts:
    if 'Verr' in c or 'verr' in c or '癫' in c:
        ver = c
        print('找到官谱:', ver)
        print('  文件:', os.listdir(ver))
# Melodiniq
p_mel = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p_mel, 'rb') as f:
    cd_mel, _ = load_chart_from_bytes(f.read())
f_mel = extract_features(cd_mel, speed=1.0)

# Verrückt IN
if ver:
    p_ver = os.path.join(ver, 'IN.json')
    if os.path.exists(p_ver):
        with open(p_ver, 'rb') as f:
            cd_ver, _ = load_chart_from_bytes(f.read())
        f_ver = extract_features(cd_ver, speed=1.0)
        print('\n=== Melodiniq vs Verrückt IN ===')
        KEYS = ['bpm','bpm_min','bpm_max','bpm_change_count','tempo_change_count','tempo_change_log_density',
                'real_notes_per_second','real_core_notes_per_second','above_avg_density_mean','eff_peak_tps_1s',
                'eff_avg_tps_1s','above_avg_duration_sec','movement_per_second','movement_density_index',
                'fast_ms_050_ratio','fast_ms_100_ratio','fast_ms_150_ratio','jack_max_run','long_jack_count',
                'chord_alternation_rate','drag_per_sec','total_notes','duration_sec','speed_volatility',
                'multi_line_sim_events','stair_speed_avg','pattern_switch_rate','rhythm_entropy',
                'density_transition_std','cross_hand_density','lane_switch_density']
        print(f'{"特征":<32}{"Melodiniq":>10}{"Verrückt":>10}')
        for k in KEYS:
            print(f'{k:<32}{f_mel.get(k,0):>10.2f}{f_ver.get(k,0):>10.2f}')
# Melodiniq BPM 原始
print('\n=== Melodiniq BPMList ===')
raw = json.load(open(p_mel, encoding='utf-8'))
bpms = raw.get('BPMList', [])
print('BPM事件数:', len(bpms))
print('BPM值:', sorted(set(b['bpm'] for b in bpms)))
print('前10:', [(b.get('bpm'), b.get('startTime')) for b in bpms[:10]])
# 24分音符: 检查音符间隔分布
from unified_parser import load_chart_from_bytes
notes = []
for jl in raw.get('judgeLineList', []):
    for n in jl.get('notes', []):
        st = n.get('startTime')
        if isinstance(st, list) and len(st) == 3:
            t = (float(st[0])*4.0 + float(st[1])*(4.0/float(st[2]))) * 8.0
            notes.append((t, n.get('type')))
notes.sort()
intervals = np.diff([t for t, _ in notes])
print(f'\n音符间隔 (ticks): P25={np.percentile(intervals,25):.1f} P50={np.percentile(intervals,50):.1f} min={intervals.min():.1f}')
# 24分 = 1拍/24 = 32/24 = 1.333 ticks; 16分 = 2 ticks
print(f'间隔<=1.34 (24分+): {np.sum(intervals<=1.34)} / {len(intervals)}')
print(f'间隔<=2.0 (16分+): {np.sum(intervals<=2.0)} / {len(intervals)}')
print('DONE')