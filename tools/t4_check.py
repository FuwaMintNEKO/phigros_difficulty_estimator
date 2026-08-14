# -*- coding: utf-8 -*-
"""验证 Melodiniq type4 是否为drag: 检查holdTime/位置/间隔"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
raw = json.load(open(p, encoding='utf-8'))
notes = []
for jl in raw.get('judgeLineList', []):
    for n in jl.get('notes', []):
        st = n.get('startTime')
        if isinstance(st, list) and len(st) == 3:
            t = (float(st[0])*4.0 + float(st[1])*(4.0/float(st[2]))) * 8.0
        else:
            t = 0
        notes.append({'type': n.get('type'), 't': t, 'x': n.get('positionX', 0),
                      'endTime': n.get('endTime'), 'speed': n.get('speed', 1.0),
                      'visibleTime': n.get('visibleTime')})
notes.sort(key=lambda n: n['t'])
t4 = [n for n in notes if n['type'] == 4]
t1 = [n for n in notes if n['type'] == 1]
print(f'type4(tap→drag?) 数量: {len(t4)}  type1: {len(t1)}')
# type4 的 endTime 检查 (drag无endTime? hold有)
has_end = sum(1 for n in t4 if n['endTime'] is not None)
print(f'type4 有endTime: {has_end}/{len(t4)}')
# type4 间隔分布 (24分?)
t4_t = np.array(sorted(n['t'] for n in t4))
if len(t4_t) > 1:
    its = np.diff(t4_t)
    print(f'type4 间隔(ticks): min={its.min():.2f} P25={np.percentile(its,25):.2f} P50={np.percentile(its,50):.2f}')
    print(f'  <=1.34(24分): {np.sum(its<=1.34)}  <=2.0(16分): {np.sum(its<=2.0)}')
# type4 位置分布 (drag通常位置连续?)
t4_x = np.array([n['x'] for n in t4])
print(f'type4 positionX: min={t4_x.min():.1f} max={t4_x.max():.1f} 唯一值数={len(set(t4_x.tolist()))}')
# type1 间隔
t1_t = np.array(sorted(n['t'] for n in t1))
if len(t1_t) > 1:
    its1 = np.diff(t1_t)
    print(f'\ntype1 间隔: min={its1.min():.2f} P25={np.percentile(its1,25):.2f} P50={np.percentile(its1,50):.2f}')
# 混合: 所有音符(非hold)间隔
print('DONE')