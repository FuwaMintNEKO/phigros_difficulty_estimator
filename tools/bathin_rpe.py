# -*- coding: utf-8 -*-
"""Bathin RPE 音符结构检查"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes

p = os.path.join(_ROOT, 'data', 'phira', 'json', '7516.json')
with open(p, 'rb') as f:
    cd, raw = load_chart_from_bytes(f.read())
lines = cd['judgeLineList']
print('judgeLine数:', len(lines))
all_notes = []
for ln in lines:
    for n in ln.get('notes', []):
        all_notes.append(n)
print('总音符:', len(all_notes))
print('音符keys样例:', list(all_notes[0].keys()) if all_notes else '无')
# speed 字段统计
has_sp = sum(1 for n in all_notes if 'speed' in n)
print('有speed字段:', has_sp)
sp = sorted(set(float(n.get('speed', 1.0)) for n in all_notes))
print('speed集合:', sp[:30])
# 长条(type=2 in RPE? 之前RPE_TYPE_MAP: RPE type2=Hold)
types = sorted(set(n.get('type') for n in all_notes))
print('type分布:', {t: sum(1 for n in all_notes if n.get('type')==t) for t in types})
holds = [n for n in all_notes if n.get('type') == 2]
print('RPE type2(hold)数:', len(holds), 'hold speed集合:', sorted(set(float(n.get('speed',1.0)) for n in holds))[:15])
# 检查unified_parser是否转换speed
import unified_parser as up
src = open(os.path.join(_ROOT, 'unified_parser.py'), encoding='utf-8').read()
import re
for m in re.finditer(r'.*speed.*', src):
    print('parser行:', m.group(0).strip()[:100])
print('DONE')