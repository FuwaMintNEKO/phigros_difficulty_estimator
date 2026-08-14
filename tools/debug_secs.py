# -*- coding: utf-8 -*-
"""调试: tick间隔 vs 秒间隔"""
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
ts_tick = times[o]; ty = types[o]; bpms = bpm_arr[o]
core_s = core[o]
tick_its = np.diff(ts_tick)
core_adj = core_s[1:] & core_s[:-1]
adj_ticks = tick_its[core_adj]
adj_bpms = bpms[1:][core_adj]
nz = adj_ticks > 0
adj_ticks = adj_ticks[nz]; adj_bpms = adj_bpms[nz]
print(f'adj_ticks: min={adj_ticks.min()} max={adj_ticks.max()} P50={np.percentile(adj_ticks,50):.0f}')
print(f'前20 ticks: {adj_ticks[:20]}')
print(f'前20 bpms: {adj_bpms[:20]}')
secs = adj_ticks / 32.0 * 60.0 / adj_bpms
print(f'前20 secs: {np.round(secs[:20]*1000,1)}ms')
thr16 = 60.0 / (adj_bpms * 16)
print(f'前20 thr16: {np.round(thr16[:20]*1000,1)}ms')
print(f'secs < thr16: {np.sum(secs < thr16)}')
# 8 tick @ 240bpm = 8/32*60/240 = 0.0625s = 62.5ms; thr16 = 60/240/16 = 15.6ms?!
print(f'\nthr16 @240bpm = 60/240/16 = {60/240/16*1000:.1f}ms ??? 16分=15.6ms?!')
print('不对! 16分音符 = 1拍/4 = 60/bpm/4 秒 = 60/(bpm*4)')
print(f'16分 @240 = 60/(240*4)*1000 = {60/(240*4)*1000:.1f}ms')
print(f'我的公式 60/(bpm*16) 错了! 应该是 60/(bpm*4)!')
print('24分 = 60/(bpm*6), 8分 = 60/(bpm*2)')
print('DONE')