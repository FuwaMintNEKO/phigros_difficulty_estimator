# -*- coding: utf-8 -*-
"""DerSchneid高仿 speedEvents 大值检查"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
p = os.path.join(DL, 'Der Schneid(1)', '1903581575578621.json')
raw = json.load(open(p, encoding='utf-8'))
vals = []
for jl in raw.get('judgeLineList', []):
    for layer in jl.get('eventLayers', []) or []:
        if layer:
            for ev in layer.get('speedEvents', []):
                vals.append(ev.get('start'))
vals = np.array(vals)
print(f'speedEvents: {len(vals)} 个')
print(f'值分布: min={vals.min():.1f} P50={np.percentile(vals,50):.1f} P90={np.percentile(vals,90):.1f} max={vals.max():.1f}')
print(f'\n>100: {np.sum(vals>100)} 个, 值: {sorted(set(vals[vals>100]))[:10]}')
print(f'9000-90000: {np.sum((vals>=9000)&(vals<90000))} 个')
print(f'\n值样例(前30): {sorted(set(vals))[:30]}')
print('DONE')