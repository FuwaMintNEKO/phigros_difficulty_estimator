# -*- coding: utf-8 -*-
"""直接用真实毫秒验证 Melodiniq 的 24分/16分 tap 间隔"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes, time_to_seconds

p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())

# collect_all_notes 的 time (已转换) 与 bpm
all_notes, jls, bpm_tl = collect_all_notes(cd)
times = np.array([n['time'] for n in all_notes])
types = np.array([n['type'] for n in all_notes])
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])

print('BPM时间线(拍):', [(round(b,1), bp) for b, bp in bpm_tl[:8]])
print('总时长:', round(t_sec.max(), 2), 's')

# tap 间隔 (真实秒)
tap = types == 1
tt = np.sort(t_sec[tap])
its = np.diff(tt)
print(f'\ntap 间隔 (真实秒):')
print(f'  min={its.min()*1000:.1f}ms P10={np.percentile(its,10)*1000:.1f}ms P25={np.percentile(its,25)*1000:.1f}ms P50={np.percentile(its,50)*1000:.1f}ms')
print(f'  <41.7ms(24分@240): {np.sum(its<0.0417)} ({np.mean(its<0.0417)*100:.0f}%)')
print(f'  41.7-62.5ms(16分@240): {np.sum((its>=0.0417)&(its<0.0625))}')
print(f'  62.5-125ms(8分@240): {np.sum((its>=0.0625)&(its<0.125))}')
# 验证: 24分音符在哪个BPM段
print('\n验证BPM: Melodiniq 240bpm时 24分=10.4ms, 16分=15.6ms')
print('  193bpm时 24分=12.95ms, 16分=19.4ms')
print(f'  <15ms: {np.sum(its<0.015)} 个 (24分@193-240)')
print(f'  15-20ms: {np.sum((its>=0.015)&(its<0.020))} 个')
print(f'  <20ms总: {np.sum(its<0.020)} 个')
# 关键: 高频区间
for lo, hi, tag in [(0, 0.02, '24分+'), (0.02, 0.042, '16-24分'), (0.042, 0.063, '16分'), (0.063, 0.125, '8分')]:
    print(f'  {tag:<10} {np.sum((its>=lo)&(its<hi))} 个 ({np.mean((its>=lo)&(its<hi))*100:.0f}%)')
print('DONE')