# -*- coding: utf-8 -*-
"""charts.json 结构 + 特殊标记探索"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'charts.json')
with open(p, encoding='utf-8') as f:
    data = json.load(f)
print('type:', type(data))
if isinstance(data, list):
    print('len:', len(data))
    print('首条:', data[0])
elif isinstance(data, dict):
    print('keys:', list(data.keys())[:10])
# 找 special 相关字段
import re
txt = json.dumps(data, ensure_ascii=False)[:200000]
for kw in ['special', 'Special', '特殊', 'Legency', 'legacy']:
    idx = txt.find(kw)
    print(f'  "{kw}" 首次出现: {"行" if idx>=0 else "无"}', idx)
print('DONE')