# -*- coding: utf-8 -*-
"""Melodiniq 音符是否带 bpm 字段"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
raw = json.load(open(p, encoding='utf-8'))
jls = raw.get('judgeLineList', [])
n0 = None
has_bpm = 0; total = 0
for jl in jls:
    for n in jl.get('notes', []):
        total += 1
        if 'bpm' in n: has_bpm += 1
        if n0 is None: n0 = n
print(f'音符总数={total}, 带bpm字段={has_bpm}')
print('音符样例keys:', list(n0.keys()) if n0 else '无')
print('DONE')