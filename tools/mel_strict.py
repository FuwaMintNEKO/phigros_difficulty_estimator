# -*- coding: utf-8 -*-
"""严格重算 Melodiniq: 找到 16分夹24分 tap 爆发段, 验证密度/位移"""
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
pos = np.array([n.get('positionX', 0) for n in all_notes])
jl = np.array([n['judge_line_idx'] for n in all_notes])
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])
dur = t_sec.max()

# tap only
tap = types == 1
tt = t_sec[tap]; tp = pos[tap]; tj = jl[tap]
o = np.argsort(tt); tt, tp, tj = tt[o], tp[o], tj[o]
print(f'tap: {len(tt)} 个, 密度 {len(tt)/dur:.2f}/s, 时长 {dur:.0f}s')

# 滑窗扫描: 找 tap 密集段 (1秒窗口)
w = 1.0
dens_arr = []
for t0 in np.arange(0, dur - w, 0.25):
    c = np.sum((tt >= t0) & (tt < t0 + w))
    dens_arr.append((t0, c))
dens_arr = np.array(dens_arr)
top = dens_arr[np.argsort(-dens_arr[:, 1])[:10]]
print('\ntap 最密1秒窗口:')
for t0, c in top:
    m = (tt >= t0) & (tt < t0 + w)
    mv = np.abs(np.diff(tp[m])).sum()
    print(f'  t={t0:.0f}s: {c:.0f} taps/s, 位移={mv:.1f}')
# 高潮段: 连续高密度段
print('\n连续 tap 密度(3秒均值) 最高的 20 秒区域:')
best = 0; bt = 0
for t0 in np.arange(0, dur - 20, 0.5):
    m = (tt >= t0) & (tt < t0 + 20)
    c = m.sum()
    if c > best: best, bt = c, t0
m = (tt >= bt) & (tt < bt + 20)
mv = np.abs(np.diff(tp[m])).sum()
print(f'  最密20秒: t={bt:.0f}-{bt+20:.0f}s, {best} taps ({best/20:.1f}/s), 位移={mv:.1f} ({mv/20:.1f}/s)')
# 这20秒的间隔分布
its = np.diff(tt[m])
print(f'  间隔: P25={np.percentile(its,25)*1000:.0f}ms P50={np.percentile(its,50)*1000:.0f}ms')
print(f'  <62.5ms(16分@240): {np.mean(its<0.0625)*100:.0f}%  <31ms(32分): {np.mean(its<0.031)*100:.0f}%')
# 16分夹24分: 相邻间隔 2:1.333 交替
print('\n交替检测 (16分=2t, 24分=1.33t):')
# 用tick间隔
tick_times = times[tap][np.argsort(tt)]
tick_its = np.diff(tick_times)
alt = 0
for i in range(1, len(tick_its)):
    a, b = tick_its[i-1], tick_its[i]
    if abs(a - 2.0) < 0.3 and abs(b - 1.333) < 0.3 or abs(a - 1.333) < 0.3 and abs(b - 2.0) < 0.3:
        alt += 1
print(f'  16分↔24分交替次数: {alt}')
print('DONE')