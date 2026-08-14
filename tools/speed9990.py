# -*- coding: utf-8 -*-
"""RPE speedEvents 9990 异常值分析"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
p = os.path.join(DL, '夢の降る日に', '5333883479687925.json')
raw = json.load(open(p, encoding='utf-8'))
all_ev = []
for jl in raw.get('judgeLineList', []):
    for layer in jl.get('eventLayers', []) or []:
        if layer:
            for ev in layer.get('speedEvents', []):
                all_ev.append((ev.get('start'), ev.get('end'), ev.get('startTime'), ev.get('endTime')))
print(f'speedEvents: {len(all_ev)} 个')
big = [e for e in all_ev if (e[0] or 0) > 100 or (e[1] or 0) > 100]
print(f'大值(>100): {len(big)} 个')
for e in big[:8]:
    print(f'  start={e[0]} end={e[1]} st={e[2]} et={e[3]}')
# 正常值分布
norm = [e[0] for e in all_ev if e[0] and e[0] <= 100]
print(f'\n正常值分布: min={min(norm):.1f} max={max(norm):.1f} P50={np.percentile(norm,50):.1f}')
# 官谱对照: 官谱speedEvents值
p2 = os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')
raw2 = json.load(open(p2, encoding='utf-8'))
off_vals = []
for jl in raw2.get('judgeLineList', []):
    for ev in jl.get('speedEvents', []):
        off_vals.append(ev.get('value'))
print(f'\n官谱speedEvents: {len(off_vals)} 个, 值范围 {min(off_vals)}~{max(off_vals)}')
print('DONE')