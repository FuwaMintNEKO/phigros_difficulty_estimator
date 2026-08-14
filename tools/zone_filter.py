# -*- coding: utf-8 -*-
"""按charts.json分区过滤: 上架(565) vs 特殊(50), 再滤整数标级"""
import os, sys, io, json, pickle, numpy as np, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
charts = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
up_ids = {c['id'] for c in charts['上架']}
sp_ids = {c['id'] for c in charts['特殊']}
print(f'上架: {len(up_ids)}  特殊: {len(sp_ids)}')
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
print('cache ranked:', len(ranked))
# cache ranked 与分区交集
in_up = [r for r in ranked if r['id'] in up_ids]
in_sp = [r for r in ranked if r['id'] in sp_ids]
neither = [r for r in ranked if r['id'] not in up_ids and r['id'] not in sp_ids]
print(f'cache ranked 在上架: {len(in_up)}, 在特殊: {len(in_sp)}, 都不在: {len(neither)}')
if neither:
    print('  都不在的谱:', [(r['id'], r['name'][:20]) for r in neither][:10])
if in_sp:
    print('  特殊区谱:', [(r['id'], r['name'][:20], round(r['diff'],1)) for r in in_sp])
print('DONE')