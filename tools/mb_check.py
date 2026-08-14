# -*- coding: utf-8 -*-
"""检查 miniburst/micro 在 FN 中的情况 + P95"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
print('FN 中 miniburst:', [f for f in app_mod.FN if 'miniburst' in f])
print('FN 中 micro:', [f for f in app_mod.FN if 'micro' in f or 'burst' in f][:20])
print('P95 中 miniburst:', [f for f in app_mod.P95 if 'miniburst' in f])
print('P95 中 micro_max:', [f for f in app_mod.P95 if 'micro' in f][:10])
# 全部FN中含burst/micro/tap_burst的
print('\nFN burst相关:', [f for f in app_mod.FN if 'burst' in f])
# 分布: ranked上架谱中 miniburst_count
import json, pickle, numpy as np
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
mb = np.array([r['feats'].get('miniburst_count', 0) for r in ranked])
print(f'\nranked miniburst_count: P50={np.median(mb):.0f} P75={np.percentile(mb,75):.0f} P90={np.percentile(mb,90):.0f} P95={np.percentile(mb,95):.0f} max={mb.max():.0f}')
mbd = np.array([r['feats'].get('miniburst_density', 0) for r in ranked])
print(f'ranked miniburst_density: P50={np.median(mbd):.3f} P75={np.percentile(mbd,75):.3f} P90={np.percentile(mbd,90):.3f} P95={np.percentile(mbd,95):.3f}')
# Melodiniq 的 percentile
mel_mb = 321; mel_mbd = 0.017
print(f'Melodiniq miniburst_count=321 → percentile {np.mean(mb<=321)*100:.0f}%')
print(f'Melodiniq miniburst_density=0.017 → percentile {np.mean(mbd<=0.017)*100:.0f}%')
print('DONE')