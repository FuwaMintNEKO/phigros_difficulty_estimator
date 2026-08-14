# -*- coding: utf-8 -*-
"""官谱 DerSchneid speedEvents 大值"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json')
raw = json.load(open(p, encoding='utf-8'))
vals = []
for jl in raw.get('judgeLineList', []):
    for ev in jl.get('speedEvents', []):
        vals.append(ev.get('value'))
vals = np.array(vals)
print(f'官谱 speedEvents: {len(vals)} 个')
print(f'值: min={vals.min():.1f} P50={np.percentile(vals,50):.1f} P90={np.percentile(vals,90):.1f} max={vals.max():.1f}')
print(f'\n大值(>10): {np.sum(vals>10)} 个, 值: {sorted(set(vals[vals>10]))[:10]}')
print(f'\n分布: {sorted(set(vals))[:25]}')
print('\n=== 官谱的499是什么? ===')
big = [ev for jl in raw.get('judgeLineList', []) for ev in jl.get('speedEvents', []) if ev.get('value', 0) > 10]
for ev in big[:5]:
    print(f'  value={ev["value"]} st={ev["startTime"]} et={ev["endTime"]}')
print('DONE')