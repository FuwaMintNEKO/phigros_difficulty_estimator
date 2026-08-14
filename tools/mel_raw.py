# -*- coding: utf-8 -*-
"""Melodiniq 原始格式与事件检查"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    head = f.read(300)
print('文件头:', repr(head[:200]))
with open(p, encoding='utf-8', errors='replace') as f:
    txt = f.read()
print('\n是JSON?', txt.strip().startswith('{'))
if txt.strip().startswith('{'):
    raw = json.loads(txt)
    print('顶层keys:', list(raw.keys())[:15])
    jls = raw.get('judgeLineList', [])
    print('线数:', len(jls))
    if jls:
        print('线0 keys:', list(jls[0].keys()))
        for k in ['judgeLineMoveEvents','judgeLineRotateEvents','judgeLineDisappearEvents','eventLayers']:
            v = jls[0].get(k)
            print(f'  线0 {k}:', type(v).__name__, len(v) if isinstance(v, list) else v)
    from collections import Counter
    evt = Counter()
    for jl in jls:
        for k in jl:
            if 'Event' in k: evt[k] += len(jl[k]) if isinstance(jl[k], list) else 1
    print('\n顶层事件统计:', dict(evt))
print('DONE')