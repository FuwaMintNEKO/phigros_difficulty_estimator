# -*- coding: utf-8 -*-
"""官谱坐标系研究: 判定线初始位置 + moveEvents坐标范围"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')
raw = json.load(open(p, encoding='utf-8'))
jls = raw.get('judgeLineList', [])
print('线数:', len(jls))
print('\n线0 完整结构:')
line0 = jls[0]
for k, v in line0.items():
    if k in ('notesAbove', 'notesBelow'): 
        print(f'  {k}: {len(v)}个音符')
    elif isinstance(v, list):
        print(f'  {k}: {len(v)}个事件, 样例={json.dumps(v[0], ensure_ascii=False)[:180] if v else "空"}')
    else:
        print(f'  {k}: {v}')
# 所有线的 positionX (如果有)
print('\n各线 positionX (初始位置):')
for i, jl in enumerate(jls[:10]):
    print(f'  线{i}: positionX={jl.get("positionX")} positionY={jl.get("positionY")} 其他位置字段={[k for k in jl if "pos" in k.lower()]}')
# moveEvents 值范围
print('\nmoveEvents start/end 值范围:')
all_start = []; all_end = []; all_start2 = []; all_end2 = []
for jl in jls:
    for ev in jl.get('judgeLineMoveEvents', []):
        all_start.append(ev.get('start')); all_end.append(ev.get('end'))
        all_start2.append(ev.get('start2')); all_end2.append(ev.get('end2'))
for nm, arr in [('start(x)', all_start), ('end(x)', all_end), ('start2(y)', all_start2), ('end2(y)', all_end2)]:
    arr = [a for a in arr if a is not None]
    if arr:
        print(f'  {nm}: min={min(arr):.3f} max={max(arr):.3f} P50={sorted(arr)[len(arr)//2]:.3f}')
# 音符 positionX 范围
all_px = []
for jl in jls:
    for n in jl.get('notesAbove', []) + jl.get('notesBelow', []):
        all_px.append(n.get('positionX', 0))
print(f'\n音符 positionX: min={min(all_px):.2f} max={max(all_px):.2f}')
print('DONE')