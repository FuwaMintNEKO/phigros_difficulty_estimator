# -*- coding: utf-8 -*-
"""核对: 原始 beat vs 转换后 time (找精度丢失点)"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes

p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
raw = json.load(open(p, encoding='utf-8'))
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())

# 原始 tap (type=1): (beat, startTime)
raw_taps = []
for jl in raw.get('judgeLineList', []):
    for n in jl.get('notes', []):
        if n.get('type') == 1:
            st = n.get('startTime')
            if isinstance(st, list) and len(st) >= 3:
                raw_taps.append((st[0] + st[1]/max(st[2],1), st, n.get('positionX', 0), '?'))
raw_taps.sort(key=lambda x: x[0])

# 转换后 tap: (time, positionX)
conv_taps = []
for jl in cd.get('judgeLineList', []):
    for n in jl.get('notesAbove', []):
        if n.get('type') == 1:
            conv_taps.append((n['time'], n.get('positionX', 0)))
conv_taps.sort(key=lambda x: x[0])

print(f'原始 tap: {len(raw_taps)}, 转换后 tap: {len(conv_taps)}')
# 对比前20个
print(f'{"#":>3}{"原始beat":>10}{"原始st":>14}{"转换time":>12}{"time/32":>10}')
for i in range(min(20, len(raw_taps), len(conv_taps))):
    rb, st, px, _ = raw_taps[i]
    ct, cpx = conv_taps[i]
    print(f'{i:>3}{rb:>10.3f}{str(st):>14}{ct:>12.0f}{ct/32:>10.3f}')
# 检查 16分间隔是否保留
print('\n=== 转换后 tap time 间隔 (tick) ===')
ct_arr = np.array([t for t, _ in conv_taps])
d = np.diff(ct_arr)
print(f'min={d.min():.2f} P10={np.percentile(d,10):.2f} P25={np.percentile(d,25):.2f} P50={np.percentile(d,50):.2f}')
print(f'  8tick(16分): {np.sum(d<=8)} 个')
print(f'  5.33tick(24分): {np.sum(d<=5.33)} 个')
print(f'  序列前30: {d[:30]}')
print('DONE')