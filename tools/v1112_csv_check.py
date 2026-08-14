# -*- coding: utf-8 -*-
"""v11.12 CSV一致性 + 最终锚点"""
import os, sys, io, pickle, numpy as np, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
with open(os.path.join(_ROOT, 'data', 'phira', 'v1112_ranked_predictions.csv'), encoding='utf-8-sig') as f:
    rows = list(csv.reader(f))[1:]
print('CSV行数:', len(rows))
# 锚点在CSV中
for row in rows:
    if row[0] in ('7516', '59064', '15875', '47264'):
        print(f"  {row[1][:20]:<22} id={row[0]} diff={row[3]} pred={row[4]} err={row[5]}")
print('app.VERSION:', app_mod.VERSION)
print('DONE')