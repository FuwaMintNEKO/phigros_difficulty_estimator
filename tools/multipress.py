# -*- coding: utf-8 -*-
"""多押结构: 0 tick 间隔的 tap 对 的真实节奏"""
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
pos = np.array([n.get('positionX', 0) for n in all_notes])
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
core = (types == 1) | (types == 3)
o = np.argsort(times)
ts = times[o]; ty = types[o]; tp = pos[o]; bpms = bpm_arr[o]; core_s = core[o]

# 多押: 相邻 core 0 tick
zero_adj = (core_s[1:] & core_s[:-1]) & (np.diff(ts) == 0)
print(f'core 0-tick 相邻对: {zero_adj.sum()}')

# 分析多押组的节奏: 组间间隔
# 找所有 core 音符, 按 tick 分组
from collections import defaultdict
grp = defaultdict(list)
for i in range(len(ts)):
    if core_s[i]:
        grp[ts[i]].append((tp[i], ty[i], bpms[i]))
# 组时间排序
times_uniq = sorted(grp.keys())
grp_its = np.diff(np.array(times_uniq))  # 组间tick间隔
print(f'\n多押组: {len(grp)} 组 (含单押)')
print(f'组间tick间隔: P25={np.percentile(grp_its,25):.0f} P50={np.percentile(grp_its,50):.0f} P75={np.percentile(grp_its,75):.0f}')
print(f'  <=8tick(16分): {np.sum(grp_its<=8)}')
print(f'  <=5.33tick(24分): {np.sum(grp_its<=5.33)}')
print(f'  序列前30: {grp_its[:30]}')
# 双押组占比
multi_groups = [t for t, notes in grp.items() if len(notes) >= 2]
print(f'\n多押组(>=2音符): {len(multi_groups)}/{len(grp)} ({len(multi_groups)/len(grp)*100:.0f}%)')
# 多押组的组间间隔
if len(multi_groups) > 1:
    mg = np.array(multi_groups)
    mg_its = np.diff(mg)
    print(f'多押组间tick间隔: P25={np.percentile(mg_its,25):.0f} P50={np.percentile(mg_its,50):.0f}')
    print(f'  <=5.33(24分): {np.sum(mg_its<=5.33)}  <=8(16分): {np.sum(mg_its<=8)}')
print('DONE')