# -*- coding: utf-8 -*-
"""multi_line_sim_events 是否在MANUAL_FLAT + 官谱分布"""
import os, sys, io, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
in_flat = [f for f, _, _ in app_mod.MANUAL_FLAT if 'multi_line' in f or 'sim' in f]
print('MANUAL_FLAT 中 multi_line/sim:', in_flat)
in_fn = [f for f in app_mod.FN if 'multi_line' in f or 'sim' in f]
print('FN 中 multi_line/sim:', in_fn)
# 官谱分布
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])
ml = np.array([r['feats'].get('multi_line_sim_events', 0) for r in ranked])
print(f'\nmulti_line_sim_events 分布: P50={np.median(ml):.1f} P75={np.percentile(ml,75):.1f} P90={np.percentile(ml,90):.1f} P95={np.percentile(ml,95):.1f}')
# 高ml且低估的谱
ps = []
from scipy.stats import spearmanr
# 看 ml 与误差
import numpy as np
# 计算当前误差需要预测——简化: 看高ml谱的diff
print('\nml>=80 的谱:')
for i in np.where(ml >= 80)[0]:
    print(f'  {ranked[i]["name"][:24]:<26} ml={ml[i]:.0f} diff={ds[i]:.1f}')
print('Feeling Blue ml=50, diff=16.7')
print('DONE')