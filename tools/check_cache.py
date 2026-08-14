# -*- coding: utf-8 -*-
"""检查cache结构 + unranked特征现状"""
import os, sys, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
print('cache keys:', list(cache.keys()))
for k, v in cache.items():
    print(f'  {k}: {len(v)} 条, 样例keys:', list(v[0].keys()) if v else '空')
# 是否有 unranked
if 'unranked' in cache:
    print('  unranked 样例:', cache['unranked'][0] if cache['unranked'] else '空')
print('DONE')