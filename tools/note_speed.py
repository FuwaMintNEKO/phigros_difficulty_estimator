# -*- coding: utf-8 -*-
"""官谱音符 speed 字段: 297个非1.0 的含义"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')
raw = json.load(open(p, encoding='utf-8'))
speeds = []
for jl in raw.get('judgeLineList', []):
    for n in jl.get('notesAbove', []) + jl.get('notesBelow', []):
        sp = n.get('speed', 1.0)
        speeds.append(sp)
speeds = np.array(speeds)
print(f'音符: {len(speeds)}, 非1.0: {np.sum(speeds != 1.0)}')
non1 = speeds[speeds != 1.0]
print(f'非1.0值: {sorted(set(non1))[:15]}')
# 分布
if len(non1):
    print(f'  min={non1.min():.3f} max={non1.max():.3f}')
print('DONE')