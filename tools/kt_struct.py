# -*- coding: utf-8 -*-
"""kyou_tags结构 + 官谱jline分布"""
import os, sys, io, json, numpy as np, pickle, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
with open(os.path.join(_ROOT, 'data', 'phira', 'kyou_tags.json'), encoding='utf-8') as f:
    kt = json.load(f)
print('type:', type(kt), 'len:', len(kt))
if isinstance(kt, list):
    print('样例:', kt[:2])
elif isinstance(kt, dict):
    k = list(kt.keys())[0]
    print('样例:', k, '->', kt[k])
print('DONE')