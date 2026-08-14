# -*- coding: utf-8 -*-
"""彻底重算: tap+hold 的 16分/24分 (tick域, 按实际BPM转秒)"""
import os, sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes, time_to_seconds

p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
all_notes, jls, bpm_tl = collect_all_notes(cd)
times = np.array([n['time'] for n in all_notes])   # tick
types = np.array([n['type'] for n in all_notes])   # 标准type: 1=tap 3=hold
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])

core = (types == 1) | (types == 3)
print(f'tap+hold: {core.sum()}')

# 按 tick 排序 (用原始tick, 不转秒)
o = np.argsort(times)
ts_tick = times[o]; ty = types[o]; bpms = bpm_arr[o]
core_s = core[o]
tick_its = np.diff(ts_tick)

# 相邻core间隔 (tick)
core_adj = core_s[1:] & core_s[:-1]
adj_ticks = tick_its[core_adj]
adj_bpms = bpms[1:][core_adj]  # 间隔用后一个音符的BPM

# 排除多押(0 tick)
nz = adj_ticks > 0
adj_ticks = adj_ticks[nz]; adj_bpms = adj_bpms[nz]

# 每个间隔的秒数: tick/32 * 60/bpm
secs = adj_ticks / 32.0 * 60.0 / adj_bpms

# 分音判定: 24分=1/6拍, 16分=1/4拍, 8分=1/2拍
# 在BPM b 下: 24分 = 60/b/24 秒, 16分 = 60/b/16 秒, 8分 = 60/b/8 秒
thr24 = 60.0 / (adj_bpms * 24)
thr16 = 60.0 / (adj_bpms * 16)
thr8 = 60.0 / (adj_bpms * 8)

n24 = np.sum(secs < thr24)
n16 = np.sum((secs >= thr24) & (secs < thr16))
n8 = np.sum((secs >= thr16) & (secs < thr8))
n_rest = np.sum(secs >= thr8)
print(f'\n=== tap+hold 相邻间隔分音 (排除多押) ===')
print(f'总间隔: {len(secs)}')
print(f'24分: {n24} ({n24/len(secs)*100:.0f}%)')
print(f'16分: {n16} ({n16/len(secs)*100:.0f}%)')
print(f'8分: {n8} ({n8/len(secs)*100:.0f}%)')
print(f'更宽: {n_rest}')

# 也统计含多押的 (不排除0)
adj_all = adj_ticks  # 已排除0
# 24分簇
print(f'\n=== 24分连打簇 ===')
if n24 > 0:
    is24 = secs < thr24
    runs = np.diff(np.concatenate(([0], is24.astype(int), [0])))
    starts = np.where(runs == 1)[0]; ends = np.where(runs == -1)[0]
    lengths = ends - starts
    print(f'24分簇数: {len(lengths)}, 最长: {lengths.max()} 连')
    from collections import Counter
    print('簇长分布:', Counter(lengths.tolist()).most_common(6))
print('DONE')