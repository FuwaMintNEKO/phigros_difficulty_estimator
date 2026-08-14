# -*- coding: utf-8 -*-
"""转换后 tap 间隔检查"""
import os, sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes

p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
jls = cd.get('judgeLineList', [])
tap_times = []
for jl in jls:
    for n in jl.get('notesAbove', []):
        if n.get('type') == 1:
            tap_times.append(n['time'])
tap_times.sort()
d = np.diff(np.array(tap_times))
print(f'tap 转换后 time 间隔(tick): min={d.min():.1f} P25={np.percentile(d,25):.1f} P50={np.percentile(d,50):.1f} P75={np.percentile(d,75):.1f}')
print(f'  16分=8tick 24分=5.33tick 8分=16tick')
print(f'  <=5.33(24分): {np.sum(d<=5.33)}')
print(f'  <=8(16分): {np.sum(d<=8)}')
print(f'  序列前30: {np.round(d[:30],1)}')
# 为什么之前算P50=125ms? 检查 time_to_seconds
print(f'\n=== time_to_seconds 验证 ===')
from feature_extractor import time_to_seconds, _parse_bpm_timeline
bpm_tl = _parse_bpm_timeline(cd)
print('bpm_timeline:', bpm_tl[:5])
# 8 tick @ 240bpm = ?
t = time_to_seconds(8, 240.0, bpm_tl)
print(f'time_to_seconds(8tick, 240bpm) = {t*1000:.1f}ms (应=62.5ms)')
t2 = time_to_seconds(16, 240.0, bpm_tl)
print(f'time_to_seconds(16tick, 240bpm) = {t2*1000:.1f}ms (应=125ms)')
print('DONE')