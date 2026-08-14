# -*- coding: utf-8 -*-
"""验证 tempo_change 计算 bug: ticks域 vs 秒域"""
import os, sys, io, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from feature_extractor import collect_all_notes, time_to_seconds
from unified_parser import load_chart_from_bytes

# 找变速官谱
p = os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
all_notes, jls, bpm_tl = collect_all_notes(cd)
times = np.array([n['time'] for n in all_notes])
intervals = np.diff(times)
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
intervals_sec = np.array([time_to_seconds(intervals[i], max(bpm_arr[i],1.0), bpm_tl) for i in range(len(intervals))])
# ticks域 tempo_change
tc_tick = 0
for i in range(1, len(intervals)):
    if intervals[i] > intervals[i-1]*1.5 or intervals[i] < intervals[i-1]*0.67:
        tc_tick += 1
# 秒域 tempo_change
tc_sec = 0
for i in range(1, len(intervals_sec)):
    if intervals_sec[i] > intervals_sec[i-1]*1.5 or intervals_sec[i] < intervals_sec[i-1]*0.67:
        tc_sec += 1
print(f'Verrückt IN: ticks域 tempo_change={tc_tick}, 秒域={tc_sec}')
# Melodiniq
p2 = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p2, 'rb') as f:
    cd2, _ = load_chart_from_bytes(f.read())
all_notes2, jls2, bpm_tl2 = collect_all_notes(cd2)
times2 = np.array([n['time'] for n in all_notes2])
intervals2 = np.diff(times2)
bpm_arr2 = np.array([n.get('bpm', 120.0) for n in all_notes2])
intervals_sec2 = np.array([time_to_seconds(intervals2[i], max(bpm_arr2[i],1.0), bpm_tl2) for i in range(len(intervals2))])
tc_tick2 = sum(1 for i in range(1,len(intervals2)) if intervals2[i]>intervals2[i-1]*1.5 or intervals2[i]<intervals2[i-1]*0.67)
tc_sec2 = sum(1 for i in range(1,len(intervals_sec2)) if intervals_sec2[i]>intervals_sec2[i-1]*1.5 or intervals_sec2[i]<intervals_sec2[i-1]*0.67)
print(f'Melodiniq: ticks域={tc_tick2}, 秒域={tc_sec2} (tempo_change_count=833特征值?)')
print('DONE')