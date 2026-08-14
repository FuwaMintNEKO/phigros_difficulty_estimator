# -*- coding: utf-8 -*-
"""验证: 同线过滤对 fast_ms 的影响 (Melodiniq 141线)"""
import os, sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from feature_extractor import collect_all_notes, time_to_seconds
from unified_parser import load_chart_from_bytes

p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
all_notes, jls, bpm_tl = collect_all_notes(cd)
times = np.array([n['time'] for n in all_notes])
jl_idx = np.array([n['judge_line_idx'] for n in all_notes])
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
positions = np.array([n.get('positionX', 0) for n in all_notes])
intervals = np.diff(times)
intervals_sec = np.array([time_to_seconds(intervals[i], max(bpm_arr[i],1.0), bpm_tl) for i in range(len(intervals))])
same_line = jl_idx[1:] == jl_idx[:-1]
print(f'总间隔: {len(intervals_sec)}, 同线: {same_line.sum()}, 跨线: {(~same_line).sum()}')
its_all = intervals_sec
its_same = intervals_sec[same_line]
print(f'\n全部间隔: <50ms={np.mean(its_all<0.05):.3f}')
print(f'同线间隔: <50ms={np.mean(its_same<0.05):.3f}  (fast_ms_050_ratio使用)')
# 24分间隔(<=13ms)的跨线/同线分布
m24 = intervals_sec < 0.013
print(f'\n<13ms(24分)间隔: {m24.sum()} 个, 其中同线={np.sum(m24 & same_line)}, 跨线={np.sum(m24 & ~same_line)}')
# 线数
print(f'\n判定线数: {len(jls)}')
# 每条线的音符数分布
from collections import Counter
lc = Counter(jl_idx.tolist())
print('线音符数: P50={} max={}'.format(np.median(list(lc.values())), max(lc.values())))
print('DONE')