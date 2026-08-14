# -*- coding: utf-8 -*-
"""彻查 core_nps: Melodiniq 音符类型分布 + core定义"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes

p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
all_notes, jls, bpm_tl = collect_all_notes(cd)
types = np.array([n['type'] for n in all_notes])
from collections import Counter
print('音符类型分布 (标准type: 1=tap 2=drag 3=hold 4=flick):')
print(Counter(types.tolist()))
# 原始RPE类型
raw = json.load(open(p, encoding='utf-8'))
raw_types = Counter()
for jl in raw.get('judgeLineList', []):
    for n in jl.get('notes', []):
        raw_types[n.get('type')] += 1
print('\n原始RPE类型分布:', dict(raw_types))
print('RPE映射: {1:1, 2:3, 3:4, 4:2} (type2=Hold→3, type3=Flick→4, type4=Drag→2)')
print('DONE')