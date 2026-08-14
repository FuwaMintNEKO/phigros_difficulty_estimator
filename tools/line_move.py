# -*- coding: utf-8 -*-
"""Melodiniq 判定线移动 vs 官谱: 谁在动"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes

def line_move_stats(path):
    with open(path, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    jls = cd.get('judgeLineList', [])
    total_moves = 0; total_notes = 0
    per_line_moves = []
    for jl in jls:
        n = len(jl.get('notesAbove', [])) + len(jl.get('notesBelow', []))
        total_notes += n
        m = len(jl.get('judgeLineMoveEvents', []))
        total_moves += m
        per_line_moves.append(m)
    return total_moves, total_notes, per_line_moves

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
for nm, p in [
    ('Melodiniq', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')),
    ('Verrückt IN', os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')),
    ('夢降日 IN', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')),
]:
    tm, tn, plm = line_move_stats(p)
    print(f'{nm}: 判定线move事件={tm}, 音符={tn}, 每线move中位数={np.median(plm):.0f}')
print('DONE')