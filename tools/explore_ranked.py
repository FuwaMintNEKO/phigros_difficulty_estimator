# -*- coding: utf-8 -*-
"""探索ranked数据结构: 找特殊谱面标记"""
import os, sys, io, pickle, numpy as np, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
print('ranked 总数:', len(ranked))
r0 = ranked[0]
print('字段:', list(r0.keys()))
# diff 分布: 整数 vs 非整数
ds = np.array([round(r['diff'], 3) for r in ranked])
int_mask = np.abs(ds - np.round(ds)) < 1e-6
print(f'\n标级为整数的谱: {int_mask.sum()}  (如 {[ (r["name"], r["diff"]) for r, m in zip(ranked, int_mask) if m][:8] })')
print(f'非整数标级: {(~int_mask).sum()}')
# level 字段样式
from collections import Counter
lv_counter = Counter(r['level'] for r in ranked)
print('\nlevel 分布(前10):', lv_counter.most_common(10))
# 特殊标记?
spec_names = [r['name'] for r in ranked if any(k in r['name'] for k in ['SP', 'Special', 'special', '愚人节', '隐藏', 'Fool'])]
print('\n特殊名谱:', len(spec_names), spec_names[:10])
# 检查 JSON 里是否有特殊标记: 找几个谱的 META
import json
for r in ranked[:5]:
    p = os.path.join(_ROOT, 'data', 'phira', 'json', f"{r['id']}.json")
    if os.path.exists(p):
        with open(p, 'rb') as f:
            head = f.read(300)
        print(f'  id={r["id"]} {r["name"][:20]} 头: {head[:80]}')
print('DONE')