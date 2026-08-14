# -*- coding: utf-8 -*-
"""验证: 转换后 Melodiniq 音符的 bpm 字段 + intervals_sec 正确性"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from feature_extractor import collect_all_notes, time_to_seconds, _parse_bpm_timeline
from unified_parser import load_chart_from_bytes

p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
all_notes, jls, bpm_tl = collect_all_notes(cd)
print('音符数:', len(all_notes))
print('bpm_timeline:', bpm_tl[:5], '... len=', len(bpm_tl))
bpms = np.array([n.get('bpm', 120.0) for n in all_notes])
print('音符bpm: min={} max={} 分布={}'.format(bpms.min(), bpms.max(), sorted(set(bpms.tolist()))[:15]))
# intervals_sec 复算
times = np.array([n['time'] for n in all_notes])
intervals = np.diff(times)
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
intervals_sec = np.array([time_to_seconds(intervals[i], max(bpm_arr[i], 1.0), bpm_tl) for i in range(len(intervals))])
its = intervals_sec
print('\nintervals_sec: <50ms={:.3f} <30ms={:.3f} <20ms={:.3f} <15ms={:.3f}'.format(
    np.mean(its<0.05), np.mean(its<0.03), np.mean(its<0.02), np.mean(its<0.015)))
print('min={:.1f}ms P25={:.1f}ms P50={:.1f}ms'.format(its.min()*1000, np.percentile(its,25)*1000, np.percentile(its,50)*1000))
# 24分音符验证: @BPM193 24分 = 60/193/24*1000 = 12.95ms
print('\n24分@193bpm = 12.95ms; 实际<13ms间隔数:', np.sum(its<0.013))
print('DONE')