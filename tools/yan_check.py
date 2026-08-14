# -*- coding: utf-8 -*-
"""RPE speedEvents 转换链检查"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
p = os.path.join(_ROOT, 'data', 'phira', 'json', '37193.json')
with open(p, 'rb') as f:
    cd, raw = load_chart_from_bytes(f.read())
# cd 里 speedEvents 结构
jls = cd.get('judgeLineList', [])
print('judgeLineList len:', len(jls))
for jl in jls:
    ses = jl.get('speedEvents', [])
    if ses:
        print('speedEvents样例:', ses[:2])
        break
# 音符
n_all = []
for jl in jls:
    n_all.extend(jl.get('notesAbove', []))
    n_all.extend(jl.get('notesBelow', []))
print('音符总数:', len(n_all))
if n_all:
    print('音符样例:', n_all[0])
# 提取特征
feats = extract_features(cd, speed=1.0)
print('\nspeed_event_count:', feats.get('speed_event_count'), 'speed_std:', feats.get('speed_std'), 'speed_max:', feats.get('speed_max'))
print('DONE')