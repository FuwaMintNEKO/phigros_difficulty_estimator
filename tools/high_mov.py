# -*- coding: utf-8 -*-
"""Melodiniq 高潮段 tap 位移: 用屏幕绝对位置(判定线位置+positionX)"""
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
pos = np.array([n.get('positionX', 0) for n in all_notes])  # 已转官方单位 (÷75)
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])
dur = t_sec.max()

# 判定线位置: Melodiniq 无 moveEvents → 所有线位置=0 → 屏幕位置=positionX
# tap only
tap = types == 1
tt = t_sec[tap]; tp = pos[tap]
o = np.argsort(tt); tt, tp = tt[o], tp[o]
print(f'tap: {len(tt)} 个')

# 高潮段 (最密20秒)
best = 0; bt = 0
for t0 in np.arange(0, dur-20, 0.5):
    m = (tt >= t0) & (tt < t0+20)
    if m.sum() > best: best, bt = m.sum(), t0
m = (tt >= bt) & (tt < bt+20)
mov = np.abs(np.diff(tp[m])).sum()
print(f'\n高潮段 t={bt:.0f}-{bt+20:.0f}s: {best} taps ({best/20:.1f}/s), 屏幕位移={mov:.1f} ({mov/20:.1f}/s)')
# 该段间隔
its = np.diff(tt[m])
print(f'  间隔: P25={np.percentile(its,25)*1000:.0f}ms P50={np.percentile(its,50)*1000:.0f}ms')
print(f'  <62.5ms(16分@240): {np.mean(its<=0.0625)*100:.0f}%')
print(f'  <41.7ms(24分@240): {np.mean(its<=0.0417)*100:.0f}%')
# 相邻tap位移分布 (全谱)
d = np.abs(np.diff(tp))
print(f'\n全谱相邻tap |Δpos|: P50={np.percentile(d,50):.1f} P90={np.percentile(d,90):.1f} 单位(官方±9)')
# 高潮段 tap 的 Δpos
d2 = np.abs(np.diff(tp[m]))
print(f'高潮段相邻tap |Δpos|: P50={np.percentile(d2,50):.1f} P90={np.percentile(d2,90):.1f}')

# 对比 Verrückt
p2 = os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')
with open(p2, 'rb') as f:
    cd2, _ = load_chart_from_bytes(f.read())
all_notes2, jls2, bpm_tl2 = collect_all_notes(cd2)
times2 = np.array([n['time'] for n in all_notes2])
types2 = np.array([n['type'] for n in all_notes2])
pos2 = np.array([n.get('positionX', 0) for n in all_notes2])
bpm_arr2 = np.array([n.get('bpm', 120.0) for n in all_notes2])
t_sec2 = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl2) for t, b in zip(times2, bpm_arr2)])
dur2 = t_sec2.max()
tap2 = types2 == 1
tt2 = t_sec2[tap2]; tp2 = pos2[tap2]
o2 = np.argsort(tt2); tt2, tp2 = tt2[o2], tp2[o2]
best2 = 0; bt2 = 0
for t0 in np.arange(0, dur2-20, 0.5):
    m2 = (tt2 >= t0) & (tt2 < t0+20)
    if m2.sum() > best2: best2, bt2 = m2.sum(), t0
m2 = (tt2 >= bt2) & (tt2 < bt2+20)
mov2 = np.abs(np.diff(tp2[m2])).sum()
print(f'\nVerrückt 高潮段 t={bt2:.0f}-{bt2+20:.0f}s: {best2} taps ({best2/20:.1f}/s), 屏幕位移={mov2:.1f} ({mov2/20:.1f}/s)')
its2 = np.diff(tt2[m2])
print(f'  间隔P50={np.percentile(its2,50)*1000:.0f}ms <62.5ms: {np.mean(its2<=0.0625)*100:.0f}%')
d3 = np.abs(np.diff(tp2[m2]))
print(f'  相邻|Δpos| P50={np.percentile(d3,50):.1f} P90={np.percentile(d3,90):.1f}')
print('DONE')