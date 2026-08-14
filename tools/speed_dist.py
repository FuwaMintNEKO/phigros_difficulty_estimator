# -*- coding: utf-8 -*-
"""官谱 vs 上架 speed特征分布 (P50/P75/P90/P95)"""
import os, sys, io, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])

KEYS = ['speed_volatility','speed_volatility_log','speed_std','speed_max','speed_range','speed_event_density','speed_event_log_density']
for k in KEYS:
    vals = np.array([r['feats'].get(k,0) for r in ranked])
    print(f'{k:<28} P50={np.median(vals):10.2f} P75={np.percentile(vals,75):10.2f} P90={np.percentile(vals,90):10.2f} P95={np.percentile(vals,95):10.2f} max={vals.max():12.1f}')
# Bathin 在分布中的位置
print()
for r in ranked:
    if r['id'] == 7516:
        print('Bathin:')
        for k in KEYS:
            v = r['feats'].get(k,0)
            vals = np.array([x['feats'].get(k,0) for x in ranked])
            pct = np.mean(vals <= v) * 100
            print(f'  {k}={v:.2f} (percentile {pct:.1f}%)')
print()
print('=== speed_volatility P95=2059 是哪些谱贡献的? ===')
vals = np.array([r['feats'].get('speed_volatility',0) for r in ranked])
for i in np.where(vals > 2059)[0]:
    print(f'  {ranked[i]["name"][:26]:<28} vol={vals[i]:.1f} diff={ds[i]:.1f}')
print('DONE')