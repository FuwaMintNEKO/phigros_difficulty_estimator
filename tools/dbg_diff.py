# -*- coding: utf-8 -*-
"""调试: tick域24分(95个) vs 秒域24分(12个) 差异根源"""
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

# tick域: <=5.33 = 24分
tick24 = adj_ticks[adj_ticks <= 5.33]
print(f'tick域24分(<=5.33tick): {len(tick24)}')
print(f'  这些间隔的tick值: {tick24[:30]}')
print(f'  对应BPM: {adj_bpms[adj_ticks <= 5.33][:30]}')
# 这些间隔的秒
secs24 = tick24 / 32.0 * 60.0 / adj_bpms[adj_ticks <= 5.33]
print(f'  对应秒: {np.round(secs24[:30]*1000, 1)}ms')
# 24分阈值 @这些BPM
thr24 = 60.0 / (adj_bpms[adj_ticks <= 5.33] * 6)
print(f'  24分阈值: {np.round(thr24[:30]*1000, 1)}ms')
print(f'  secs < thr24: {np.sum(secs24 < thr24)}')
print(f'\n=== 结论 ===')
print(f'tick域24分 {len(tick24)} 个, 但按BPM换算秒后只有 {np.sum(secs24 < thr24)} 个是真24分')
print(f'差异: tick<=5.33 在低BPM段(193) = {5.33/32*60/193*1000:.1f}ms, 而193bpm的24分={60/(193*6)*1000:.1f}ms')
print('DONE')