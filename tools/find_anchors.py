# -*- coding: utf-8 -*-
"""查ranked cache里的 59064/15875"""
import os, sys, pickle, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
for group in ('ranked', 'unranked'):
    for r in cache.get(group, []):
        if r['id'] in (59064, 15875, 41242, 28438):
            print(group, r['id'], r['name'], r['level'], 'diff=', r.get('diff'), 'charter=', r.get('charter', ''))
print('DONE')