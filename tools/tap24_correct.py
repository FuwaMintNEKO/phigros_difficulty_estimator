# -*- coding: utf-8 -*-
"""修正公式后: tap+hold 分音精确统计 (16分=60/(bpm*4), 24分=60/(bpm*6))"""
import os, sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes

p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
all_notes, jls, bpm_tl = collect_all_notes(cd)
times = np.array([n['time'] for n in all_notes])
types = np.array([n['type'] for n in all_notes])
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
core = (types == 1) | (types == 3)
o = np.argsort(times)
ts_tick = times[o]; bpms = bpm_arr[o]; core_s = core[o]
tick_its = np.diff(ts_tick)
core_adj = core_s[1:] & core_s[:-1]
adj_ticks = tick_its[core_adj]
adj_bpms = bpms[1:][core_adj]
nz = adj_ticks > 0
adj_ticks = adj_ticks[nz]; adj_bpms = adj_bpms[nz]
secs = adj_ticks / 32.0 * 60.0 / adj_bpms

# 正确阈值
thr24 = 60.0 / (adj_bpms * 6)    # 24分
thr16 = 60.0 / (adj_bpms * 4)    # 16分
thr8 = 60.0 / (adj_bpms * 2)     # 8分
n24 = np.sum(secs < thr24)
n16 = np.sum((secs >= thr24) & (secs < thr16))
n8 = np.sum((secs >= thr16) & (secs < thr8))
n_rest = np.sum(secs >= thr8)
tot = len(secs)
print(f'=== tap+hold 分音统计 (排除多押) ===')
print(f'总相邻core间隔: {tot}')
print(f'24分 (间隔<{np.median(thr24)*1000:.0f}ms): {n24} ({n24/tot*100:.1f}%)')
print(f'16分: {n16} ({n16/tot*100:.1f}%)')
print(f'8分: {n8} ({n8/tot*100:.1f}%)')
print(f'更宽: {n_rest} ({n_rest/tot*100:.1f}%)')
print(f'\n16分+24分合计: {n16+n24} ({ (n16+n24)/tot*100:.1f}%)')

# 24分簇
print(f'\n=== 24分连打簇 ===')
if n24 > 0:
    is24 = secs < thr24
    runs = np.diff(np.concatenate(([0], is24.astype(int), [0])))
    starts = np.where(runs == 1)[0]; ends = np.where(runs == -1)[0]
    lengths = ends - starts
    from collections import Counter
    print(f'簇数: {len(lengths)}, 最长: {lengths.max()} 连')
    print('分布:', Counter(lengths.tolist()).most_common(8))
print('DONE')