# -*- coding: utf-8 -*-
"""Melodiniq 高潮段 tap 详细: 展示真实分音构成"""
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
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])
o = np.argsort(t_sec)
ts = t_sec[o]; ty = types[o]

# 高潮段 106-126s
m = (ts >= 106) & (ts < 126)
seg = ts[m]; seg_ty = ty[m]
its = np.diff(seg)
print(f'高潮段(106-126s): {len(seg)} 音符')
print(f'\n按类型: tap={np.sum(seg_ty==1)} drag={np.sum(seg_ty==2)} hold={np.sum(seg_ty==3)} flick={np.sum(seg_ty==4)}')
# tap+hold 间隔
core_adj = ((seg_ty[1:]==1)|(seg_ty[1:]==3)) & ((seg_ty[:-1]==1)|(seg_ty[:-1]==3)) & (its>1e-6)
cits = its[core_adj]
print(f'\ntap+hold相邻间隔: {len(cits)} 个')
print(f'  24分(<42ms): {np.sum(cits<0.042)} 个')
print(f'  16分(42-63ms): {np.sum((cits>=0.042)&(cits<0.063))} 个')
print(f'  8分(63-125ms): {np.sum((cits>=0.063)&(cits<0.125))} 个')
# 含drag的间隔
print(f'\n含drag相邻间隔: {len(its[its>1e-6])} 个')
print(f'  <42ms: {np.sum(its[its>1e-6]<0.042)} 个')
# 多押(0ms)
print(f'\n多押(0ms): {np.sum(its==0)} 个 (同时音符)')
print('DONE')