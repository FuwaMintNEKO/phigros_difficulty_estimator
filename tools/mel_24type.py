# -*- coding: utf-8 -*-
"""最终确认: Melodiniq 的24分是否全是drag + RPEVersion"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
raw = json.load(open(p, encoding='utf-8'))
meta = raw.get('META', {})
print('RPEVersion:', meta.get('RPEVersion'))
print('曲名:', meta.get('name'), '| 谱师:', meta.get('charter'), '| 定数:', meta.get('difficulty'))
# 24分间隔 按类型
notes = []
for jl in raw.get('judgeLineList', []):
    for n in jl.get('notes', []):
        st = n.get('startTime')
        if isinstance(st, list) and len(st) == 3:
            t = (float(st[0])*4.0 + float(st[1])*(4.0/float(st[2]))) * 8.0
        else:
            t = 0
        notes.append({'type': n.get('type'), 't': t})
notes.sort(key=lambda n: n['t'])
ts = np.array([n['t'] for n in notes])
tys = np.array([n['type'] for n in notes])
its = np.diff(ts)
# 24分(<=1.34tick) 的类型对
pairs = [(tys[i], tys[i+1]) for i in range(len(its)) if its[i] <= 1.34]
from collections import Counter
print(f'\n24分间隔总数: {len(its[its<=1.34])}, 按相邻类型:')
print(Counter(pairs).most_common(10))
print('\n含义: RPE type4=Drag(黄键, 零操作), type1=Tap')
tap24 = sum(1 for a,b in pairs if a==1 and b==1)
print(f'tap-tap 24分: {tap24} (真手指爆发)')
print('DONE')