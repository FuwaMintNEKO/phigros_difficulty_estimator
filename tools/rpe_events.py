# -*- coding: utf-8 -*-
"""RPE事件结构: 高仿夢降日的 eventLayers/顶层事件"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'tools', '_tmp_dl_charts', '夢の降る日に', '5333883479687925.json')
with open(p, encoding='utf-8') as f:
    raw = json.load(f)
jls = raw.get('judgeLineList', [])
print('线数:', len(jls))
for i, jl in enumerate(jls[:3]):
    print(f'\n线{i}: keys={list(jl.keys())}')
    for k in ['judgeLineMoveEvents', 'judgeLineRotateEvents', 'judgeLineDisappearEvents', 'eventLayers', 'extended']:
        v = jl.get(k)
        if v:
            print(f'  {k}: {type(v).__name__} len={len(v) if hasattr(v,"__len__") else "?"}')
            if isinstance(v, list) and v:
                print(f'    样例: {json.dumps(v[0], ensure_ascii=False)[:200]}')
            elif isinstance(v, dict):
                print(f'    keys: {list(v.keys())[:8]}')
print('\n=== 统计所有线 ===')
from collections import Counter
cnt = Counter()
for jl in jls:
    for k in jl:
        if 'Event' in k or 'event' in k or k == 'extended':
            cnt[k] += 1
print(dict(cnt))
print('DONE')