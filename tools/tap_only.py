# -*- coding: utf-8 -*-
"""Melodiniq: tap-only 分析 (去掉drag后) 尾杀密度"""
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
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])
dur = t_sec.max()

# tap+hold (core)
core_mask = (types == 1) | (types == 3)
tap_mask = types == 1
print(f'总音符={len(times)} tap={tap_mask.sum()} hold={(types==3).sum()} drag={(types==2).sum()} flick={(types==4).sum()}')

# 全程 tap 密度
tc = t_sec[tap_mask]
print(f'tap密度: {len(tc)/dur:.2f}/s')
# 尾杀 tap
tail = tc >= dur * 0.8
print(f'尾20% tap: {tail.sum()} 个, 密度={tail.sum()/(dur*0.2):.2f}/s')
# tap-only 间隔 (秒)
tc_sorted = np.sort(tc)
its = np.diff(tc_sorted)
print(f'tap间隔: P25={np.percentile(its,25)*1000:.0f}ms P50={np.percentile(its,50)*1000:.0f}ms <50ms={np.mean(its<0.05)*100:.0f}%')
# 最密10秒 tap
best = 0; best_t = 0
for t0 in np.arange(0, dur-10, 1.0):
    c = np.sum((tc >= t0) & (tc < t0+10))
    if c > best: best, best_t = c, t0
print(f'\n最密10秒 tap: {best} 个 ({best/10:.1f}/s) @ t={best_t:.0f}s')
# 最密10秒 tap的位移
m = (tc >= best_t) & (tc < best_t+10)
idx = np.argsort(tc[m])
mv = np.abs(np.diff(pos[tap_mask][np.argsort(tc)][np.where(m)[0]][idx])).sum() if False else None
# 简化: tap 位置序列
tap_pos_sorted = pos[tap_mask][np.argsort(tc)]
tap_t_sorted = tc[np.argsort(tc)]
m2 = (tap_t_sorted >= best_t) & (tap_t_sorted < best_t+10)
mv2 = np.abs(np.diff(tap_pos_sorted[m2])).sum()
print(f'  该段tap位移: {mv2:.1f} ({mv2/10:.1f}/s)')
print('DONE')