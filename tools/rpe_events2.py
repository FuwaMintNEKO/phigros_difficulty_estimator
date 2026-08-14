# -*- coding: utf-8 -*-
"""eventLayers 完整结构: 找 move/rotate 事件"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'tools', '_tmp_dl_charts', '夢の降る日に', '5333883479687925.json')
with open(p, encoding='utf-8') as f:
    raw = json.load(f)
jls = raw.get('judgeLineList', [])
# 统计所有 eventLayers 里的事件类型
from collections import Counter
evt_types = Counter()
for jl in jls:
    for layer in jl.get('eventLayers', []):
        if layer is None: continue
        for k in layer:
            if k.endswith('Events') or k.endswith('events'):
                evt_types[k] += 1
print('事件类型统计:', dict(evt_types))
# 看 move/rotate 相关
for jl in jls[:1]:
    for li, layer in enumerate(jl.get('eventLayers', [])):
        if layer is None: continue
        for k, v in layer.items():
            if 'move' in k.lower() or 'rotate' in k.lower() or 'position' in k.lower() or 'pos' in k.lower():
                print(f'线0 layer{li} {k}: len={len(v) if isinstance(v, list) else "?"}')
                if isinstance(v, list) and v:
                    print(f'  样例: {json.dumps(v[0], ensure_ascii=False)[:250]}')
# 官谱 move 事件结构参考
p2 = os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')
with open(p2, encoding='utf-8') as f:
    off = json.load(f)
for jl in off.get('judgeLineList', []):
    if jl.get('judgeLineMoveEvents'):
        print('\n官谱move样例:', json.dumps(jl['judgeLineMoveEvents'][0], ensure_ascii=False))
        print('官谱move keys:', list(jl['judgeLineMoveEvents'][0].keys()))
        break
print('DONE')