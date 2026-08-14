# -*- coding: utf-8 -*-
"""全音符(含drag) 24分检测 + 同线/跨线 + 高潮段聚焦"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes, time_to_seconds

p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
all_notes, jls, bpm_tl = collect_all_notes(cd)
times = np.array([n['time'] for n in all_notes])
types = np.array([n['type'] for n in all_notes])
jl_idx = np.array([n['judge_line_idx'] for n in all_notes])
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])

# 全音符间隔
o = np.argsort(t_sec)
ts_sorted = t_sec[o]; ty_sorted = types[o]; jl_sorted = jl_idx[o]
its = np.diff(ts_sorted)
print('全音符(1786)间隔:')
print(f'  min={its.min()*1000:.1f}ms P10={np.percentile(its,10)*1000:.1f} P25={np.percentile(its,25)*1000:.1f} P50={np.percentile(its,50)*1000:.1f}')
for lo, hi, tag in [(0, 0.02, '24分+'), (0.02, 0.0313, '32分@240'), (0.0313, 0.042, '24分@240'), (0.042, 0.063, '16分@240')]:
    print(f'  {tag:<12} {np.sum((its>=lo)&(its<hi))} ({np.mean((its>=lo)&(its<hi))*100:.0f}%)')
# 高潮段 (最密20秒)
dur = ts_sorted.max()
best = 0; bt = 0
for t0 in np.arange(0, dur-20, 0.5):
    c = np.sum((ts_sorted >= t0) & (ts_sorted < t0+20))
    if c > best: best, bt = c, t0
m = (ts_sorted >= bt) & (ts_sorted < bt+20)
print(f'\n高潮段 t={bt:.0f}-{bt+20:.0f}s: {best} 音符 ({best/20:.1f}/s)')
seg_its = np.diff(ts_sorted[m])
for lo, hi, tag in [(0, 0.02, '24分+'), (0.02, 0.042, '24分'), (0.042, 0.063, '16分'), (0.063, 0.125, '8分')]:
    print(f'  {tag:<12} {np.sum((seg_its>=lo)&(seg_its<hi))} ({np.mean((seg_its>=lo)&(seg_its<hi))*100:.0f}%)')
# 高潮段 24分的类型
print('\n高潮段 24分(<42ms) 类型组合:')
from collections import Counter
pairs = Counter()
for i in range(1, len(ts_sorted)):
    if m[i-1] and m[i] and ts_sorted[i]-ts_sorted[i-1] < 0.042:
        pairs[(int(ty_sorted[i-1]), int(ty_sorted[i]))] += 1
print(pairs.most_common(10))
print('(1=tap 2=drag 3=hold 4=flick)')
print('DONE')