# -*- coding: utf-8 -*-
"""Melodiniq: 纯 tap+hold 的 24 分间隔精确统计"""
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

times = np.array([n['time'] for n in all_notes])   # ticks
types = np.array([n['type'] for n in all_notes])   # 标准: 1=tap 2=drag 3=hold 4=flick
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])

core = (types == 1) | (types == 3)
print(f'总音符={len(times)} tap+hold={core.sum()} (tap={np.sum(types==1)} hold={np.sum(types==3)})')

# 相邻的 tap+hold 对 (时间排序后, 相邻两个都是core)
o = np.argsort(t_sec)
ts = t_sec[o]; ty = types[o]
core_sorted = core[o]
its = np.diff(ts)
core_adj = core_sorted[1:] & core_sorted[:-1]

# 相邻core间隔(排除0多押)
adj_its = its[core_adj]
non_zero = adj_its[adj_its > 1e-6]
print(f'\n相邻core(tap+hold)间隔: {len(adj_its)} 个, 排除多押0后: {len(non_zero)} 个')

# 24分定义: 按每个音符实际BPM
# 每个间隔 i 对应音符 i 的BPM
bpm_sorted = bpm_arr[o]
# 间隔 i 的24分阈值 = 60/(bpm*24) 秒
thr24 = 60.0 / (bpm_sorted[1:][core_adj][non_zero_idx()] * 24) if False else None

# 直接用毫秒阈值 (BPM 193-240: 24分 = 12.95ms ~ 10.4ms)
# 用每个音符的BPM精确算
core_idx = np.where(core_adj)[0]  # 间隔在sorted数组中的位置
bpm_of_interval = bpm_sorted[core_idx + 1]  # 用后一个音符的BPM
thr24_arr = 60.0 / (bpm_of_interval * 24.0)
thr16_arr = 60.0 / (bpm_of_interval * 16.0)

n24 = np.sum((non_zero < thr24_arr[np.where(core_adj)[0][adj_its > 1e-6]]) & (adj_its > 1e-6)) if False else None
# 重新算: 非零间隔对应的阈值
nz_mask = adj_its > 1e-6
nz_its = adj_its[nz_mask]
nz_thr24 = thr24_arr[nz_mask]
nz_thr16 = thr16_arr[nz_mask]

n24 = np.sum(nz_its < nz_thr24)
n16_24 = np.sum((nz_its >= nz_thr24) & (nz_its < nz_thr16))
print(f'\n=== 纯 tap+hold 分音统计 (按实际BPM) ===')
print(f'24分 (间隔 < {nz_thr24.min()*1000:.1f}~{nz_thr24.max()*1000:.1f}ms): {n24} 个')
print(f'16分 (24分~16分之间): {n16_24} 个')
print(f'8分+: {len(nz_its) - n24 - n16_24} 个')

# 24分簇: 连续3+个24分间隔
print(f'\n=== 24分连打簇 (连续>=3个24分间隔) ===')
if n24 > 0:
    is24 = nz_its < nz_thr24
    runs = np.diff(np.concatenate(([0], is24.astype(int), [0])))
    starts = np.where(runs == 1)[0]
    ends = np.where(runs == -1)[0]
    lengths = ends - starts
    for l in sorted(set(lengths), reverse=True):
        cnt = np.sum(lengths == l)
        print(f'  连续{l}个24分: {cnt} 簇')
    # 最大簇的位置
    mi = np.argmax(lengths)
    t0 = nz_its[:starts[mi]].sum() if starts[mi] > 0 else 0
    print(f'\n最大24分簇: {lengths[mi]} 连, 起始时间≈{t0:.1f}s')

# 高潮段检查 (106-126s)
m = (ts >= 106) & (ts < 126)
print(f'\n=== 高潮段(106-126s) tap+hold ===')
seg_ty = ty[m]
print(f'  该段音符: {len(seg_ty)} (tap={np.sum(seg_ty==1)} drag={np.sum(seg_ty==2)} hold={np.sum(seg_ty==3)} flick={np.sum(seg_ty==4)})')
seg_core = (seg_ty == 1) | (seg_ty == 3)
seg_ts = ts[m]
seg_its = np.diff(seg_ts)
seg_adj = seg_core[1:] & seg_core[:-1]
seg_adj_its = seg_its[seg_adj]
seg_adj_its = seg_adj_its[seg_adj_its > 1e-6]
print(f'  相邻core间隔: {len(seg_adj_its)} 个')
print(f'  间隔分布: P25={np.percentile(seg_adj_its,25)*1000:.0f}ms P50={np.percentile(seg_adj_its,50)*1000:.0f}ms P75={np.percentile(seg_adj_its,75)*1000:.0f}ms')
print(f'  <20ms: {np.sum(seg_adj_its<0.020)} 个')
print(f'  20-42ms: {np.sum((seg_adj_its>=0.020)&(seg_adj_its<0.042))} 个')
print(f'  42-63ms: {np.sum((seg_adj_its>=0.042)&(seg_adj_its<0.063))} 个')
print('DONE')