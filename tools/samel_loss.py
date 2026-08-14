# -*- coding: utf-8 -*-
"""同线过滤 vs 全量: 24分/16分 的捕获率 (Melodiniq高潮段)"""
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
jl_idx = np.array([n['judge_line_idx'] for n in all_notes])
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])
o = np.argsort(t_sec)
ts_sorted = t_sec[o]; jl_sorted = jl_idx[o]
its = np.diff(ts_sorted)
same = jl_sorted[1:] == jl_sorted[:-1]

# 高潮段
dur = ts_sorted.max()
best = 0; bt = 0
for t0 in np.arange(0, dur-20, 0.5):
    c = np.sum((ts_sorted >= t0) & (ts_sorted < t0+20))
    if c > best: best, bt = c, t0
m = (ts_sorted >= bt) & (ts_sorted < bt+20)
m_pair = m[1:] & m[:-1]

print('高潮段 24分(<42ms) 间隔:')
m24 = m_pair & (its < 0.042)
print(f'  总数: {m24.sum()}')
print(f'  同线: {np.sum(m24 & same)} ({np.mean(m24 & same)*100:.0f}%)')
print(f'  跨线: {np.sum(m24 & ~same)} ({np.mean(m24 & ~same)*100:.0f}%)')
print('\n高潮段 16分(42-63ms) 间隔:')
m16 = m_pair & (its >= 0.042) & (its < 0.063)
print(f'  总数: {m16.sum()}')
print(f'  同线: {np.sum(m16 & same)} ({np.mean(m16 & same)*100:.0f}%)')
print(f'  跨线: {np.sum(m16 & ~same)} ({np.mean(m16 & ~same)*100:.0f}%)')
print('\n=== 结论: 同线过滤丢了多少? ===')
print(f'24分: 同线捕获 {np.sum(m24&same)}/{m24.sum()} = {np.sum(m24&same)/max(m24.sum(),1)*100:.0f}%')
print(f'16分: 同线捕获 {np.sum(m16&same)}/{m16.sum()} = {np.sum(m16&same)/max(m16.sum(),1)*100:.0f}%')
print('DONE')