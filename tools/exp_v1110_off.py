# -*- coding: utf-8 -*-
"""16.5+段: 官谱识别 + kyou_type + err构成"""
import os, sys, numpy as np, io, pickle, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])
# 官谱名单: 尝试从kyou_tags或cache读取
try:
    with open(os.path.join(_ROOT, 'data', 'kyou_tags.json'), encoding='utf-8') as f:
        kt = json.load(f)
    print('kyou_tags.json:', len(kt), '首')
    # 看结构
    if kt:
        k = list(kt.keys())[0]
        print('sample key:', k, '->', kt[k] if not isinstance(kt[k], dict) else list(kt[k].keys()))
except Exception as e:
    print('kyou_tags err', e)
# 检查cache里ranked有无 is_official 字段
r0 = ranked[0]
print('ranked[0] keys:', list(r0.keys()))
print('ranked[0] sample:', {k: r0[k] for k in ['name','diff','level'] if k in r0})
print('DONE')