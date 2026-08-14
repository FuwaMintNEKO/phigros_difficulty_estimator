# -*- coding: utf-8 -*-
"""超短间隔(32分)定位: 是否真32分 还是 tick精度伪影"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes, time_to_seconds

p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
all_notes, jls, bpm_tl = collect_all_notes(cd)
times = np.array([n['time'] for n in all_notes])
types = np.array([n['type'] for n in all_notes])
pos = np.array([n.get('positionX', 0) for n in all_notes])
jl = np.array([n['judge_line_idx'] for n in all_notes])
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])

# 所有音符的超短间隔 (<31.3ms)
for i in range(1, len(t_sec)):
    d = t_sec[i] - t_sec[i-1]
    if d < 0.0313 and d > 0:
        print(f't={t_sec[i]:.2f}s: 间隔={d*1000:.1f}ms type={types[i]}(前{types[i-1]}) 线={jl[i]}x{jl[i-1]} pos={pos[i]:.1f}x{pos[i-1]:.1f}')
# 统计超短间隔的类型组合
short = []
for i in range(1, len(t_sec)):
    d = t_sec[i] - t_sec[i-1]
    if d < 0.0313 and d > 0:
        short.append((int(types[i-1]), int(types[i]), d*1000, jl[i-1], jl[i], pos[i-1], pos[i]))
print(f'\n超短间隔总数: {len(short)}')
from collections import Counter
print('类型组合:', Counter((a,b) for a,b,_,_,_,_,_ in short).most_common(8))
print('同线占比:', sum(1 for s in short if s[3]==s[4])/len(short))
print('\n样例(前10):')
for s in short[:10]:
    print(f'  type {s[0]}→{s[1]} {s[2]:.1f}ms 线{s[3]}→{s[4]} pos {s[5]:.1f}→{s[6]:.1f}')
print('DONE')