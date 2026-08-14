# -*- coding: utf-8 -*-
"""first_note_time 差异: 高仿608 vs 官谱192 (时间基准)"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
p1 = os.path.join(DL, '夢の降る日に', '5333883479687925.json')
p2 = os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')
with open(p1, 'rb') as f:
    cd1, _ = load_chart_from_bytes(f.read())
with open(p2, 'rb') as f:
    cd2, _ = load_chart_from_bytes(f.read())
n1 = collect_all_notes(cd1)[0]
n2 = collect_all_notes(cd2)[0]
t1 = np.array([n['time'] for n in n1])
t2 = np.array([n['time'] for n in n2])
print(f'高仿: 音符={len(t1)}, 首音={t1.min():.1f} tick, 尾音={t1.max():.1f}')
print(f'官谱: 音符={len(t2)}, 首音={t2.min():.1f} tick, 尾音={t2.max():.1f}')
print(f'\n高仿首音608tick = 19拍, 官谱192tick = 6拍')
# 高仿为什么首音在19拍? 看原始RPE
raw = json.load(open(p1, encoding='utf-8'))
first_st = None
for jl in raw.get('judgeLineList', []):
    for n in jl.get('notes', []):
        st = n.get('startTime')
        if isinstance(st, list) and len(st) >= 3:
            beat = st[0] + st[1]/max(st[2],1)
            if first_st is None or beat < first_st:
                first_st = beat
print(f'高仿原始首音 beat={first_st}')
# 官谱原始
raw2 = json.load(open(p2, encoding='utf-8'))
first2 = None
for jl in raw2.get('judgeLineList', []):
    for n in jl.get('notesAbove', []) + jl.get('notesBelow', []):
        t = n.get('time', 0)
        if first2 is None or t < first2:
            first2 = t
print(f'官谱原始首音 tick={first2} (= {first2/32:.1f}拍)')
print('\n高仿608tick=19拍 vs 官谱192tick=6拍 → 高仿首音晚了13拍?')
print('可能高仿谱有前置空拍 (offset差异)')
print('DONE')