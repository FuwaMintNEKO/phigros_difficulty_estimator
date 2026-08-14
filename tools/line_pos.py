# -*- coding: utf-8 -*-
"""Melodiniq 141条线的位置定义: 判定线如何在屏幕上分布"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes

p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
jls = cd.get('judgeLineList', [])
print('线数:', len(jls))
# 每条线的音符 positionX 范围
print(f'{"线":>4}{"音符数":>6}{"posX范围":>16}{"mean":>8}')
for li, jl in enumerate(jls[:15]):
    notes = jl.get('notesAbove', []) + jl.get('notesBelow', [])
    if notes:
        px = np.array([n.get('positionX', 0) for n in notes])
        print(f'{li:>4}{len(notes):>6}{f"{px.min():.1f}~{px.max():.1f}":>16}{px.mean():>8.1f}')
# 所有线的 posX 均值分布 (线的"位置"特征)
means = []
for jl in jls:
    notes = jl.get('notesAbove', []) + jl.get('notesBelow', [])
    if notes:
        means.append(np.mean([n.get('positionX', 0) for n in notes]))
means = np.array(means)
print(f'\n各线音符posX均值: min={means.min():.1f} max={means.max():.1f} std={means.std():.1f}')
print('排序后前20:', np.sort(means)[:20].round(1))
# RPE原始: 判定线的Group/father/x?
raw = json.load(open(p, encoding='utf-8'))
print('\nRPE原始线字段:')
line0 = raw['judgeLineList'][0]
for k in ['Group','Name','father','group','alphaControl','posControl','yControl','x','y','positionX']:
    if k in line0:
        print(f'  {k}: {line0[k]}')
print('DONE')