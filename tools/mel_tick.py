# -*- coding: utf-8 -*-
"""RPE时间单位验证: Melodiniq 高潮段 tap 间隔的真实值"""
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
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])

tap = types == 1
tt = t_sec[tap]
o = np.argsort(tt); tt = tt[o]
# 高潮段 t=142-162s
m = (tt >= 142) & (tt < 162)
seg = tt[m]
print(f'高潮段 tap: {len(seg)} 个')
its = np.diff(seg)
print(f'间隔: min={its.min()*1000:.1f}ms P25={np.percentile(its,25)*1000:.1f}ms P50={np.percentile(its,50)*1000:.1f}ms P75={np.percentile(its,75)*1000:.1f}ms max={its.max()*1000:.1f}ms')
# BPM 240: 16分=62.5ms, 24分=41.7ms, 32分=31.25ms
print(f'\nBPM240下: 16分=62.5ms 24分=41.7ms 32分=31.3ms')
print(f'<41.7ms(24分+): {np.sum(its<0.0417)} ({np.mean(its<0.0417)*100:.0f}%)')
print(f'41.7-62.5ms(16分): {np.sum((its>=0.0417)&(its<0.0625))}')
print(f'<31.3ms(32分): {np.sum(its<0.0313)}')
# 检查原始RPE 时间转换: startTime [m,b,d] → 我的公式 (m*4+b*4/d)*8
# 看高潮段音符的原始 startTime
raw = json.load(open(p, encoding='utf-8'))
all_raw = []
for jl in raw.get('judgeLineList', []):
    for n in jl.get('notes', []):
        st = n.get('startTime')
        if isinstance(st, list) and len(st) == 3:
            all_raw.append((st, n.get('type')))
# 找间隔最小的几对
tick_vals = []
for st, ty in all_raw:
    if isinstance(st, list) and len(st) == 3:
        tick_vals.append(((st[0]*4.0 + st[1]*(4.0/st[2]))*8.0, ty))
tick_vals.sort()
tvs = np.array([v[0] for v in tick_vals])
tys = np.array([v[1] for v in tick_vals])
tick_its = np.diff(tvs)
print(f'\n全部音符tick间隔分布: min={tick_its.min():.2f} P25={np.percentile(tick_its,25):.2f} P50={np.percentile(tick_its,50):.2f}')
print(f'  间隔<=1.34: {np.sum(tick_its<=1.34)}  <=2.0: {np.sum(tick_its<=2.0)}')
# 1拍=32tick? 那16分=2tick, 24分=1.333tick
# 但RPE的beat是多少tick? 检查: 看BPM240段的音符间隔
print('DONE')