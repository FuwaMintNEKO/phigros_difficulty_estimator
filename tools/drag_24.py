# -*- coding: utf-8 -*-
"""Melodiniq 高潮段 240 drag 的 24分细节: 确认它们是不是'密集蓝键'错觉来源"""
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
o = np.argsort(t_sec)
ts, ty, tp = t_sec[o], types[o], pos[o]

# 高潮段 106-126s
m = (ts >= 106) & (ts < 126)
seg_ts, seg_ty, seg_tp = ts[m], ty[m], tp[m]
print('高潮段(106-126s) 按类型统计:')
for t in [1, 2, 3, 4]:
    n = np.sum(seg_ty == t)
    names = {1:'tap蓝', 2:'drag黄', 3:'hold长条', 4:'flick红'}
    print(f'  {names[t]}: {n}')
# drag 的位置分布
d = seg_ty == 2
print(f'\ndrag 位置: min={seg_tp[d].min():.1f} max={seg_tp[d].max():.1f} P50={np.median(seg_tp[d]):.1f}')
print(f'tap 位置: min={seg_tp[~d].min():.1f} max={seg_tp[~d].max():.1f} P50={np.median(seg_tp[~d]):.1f}')
# drag 与 tap 的交替模式
print('\n相邻音符类型序列 (前30):')
seq = [names_map[t][0] for t in seg_ty[:30]] if False else ''.join({1:'B',2:'Y',3:'H',4:'R'}[t] for t in seg_ty[:60])
print(f'  {seq}')
print('  (B=蓝tap Y=黄drag H=长条 R=红flick)')
# 24分间隔的类型
its = np.diff(seg_ts)
pairs24 = []
for i in range(len(its)):
    if its[i] < 0.042 and its[i] > 1e-6:
        pairs24.append((seg_ty[i], seg_ty[i+1]))
from collections import Counter
print(f'\n24分间隔({len(pairs24)}个)类型:')
print(Counter(pairs24).most_common(8))
print('DONE')