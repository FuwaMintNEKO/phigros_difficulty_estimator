# -*- coding: utf-8 -*-
"""miniburst_count 设计: 官谱15+ vs 全谱 分布, 找安全阈值"""
import os, sys, io, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = cache['official']
off15 = [o for o in official if o['diff'] >= 15]
# miniburst_count (间隔<0.0625拍 = 16分/拍内)
mb = np.array([o['feats'].get('miniburst_count', 0) for o in official])
mb15 = np.array([o['feats'].get('miniburst_count', 0) for o in off15])
y15 = np.array([o['diff'] for o in off15])
print(f'miniburst_count 全官谱: P75={np.percentile(mb,75):.0f} P90={np.percentile(mb,90):.0f} P95={np.percentile(mb,95):.0f}')
print(f'miniburst_count 官谱15+: P50={np.percentile(mb15,50):.0f} P75={np.percentile(mb15,75):.0f} P90={np.percentile(mb15,90):.0f} P95={np.percentile(mb15,95):.0f}')
print(f'Melodiniq=321')
# 15+官谱中 mb>=321 的定数分布
mk = mb15 >= 321
print(f'15+官谱 mb>=321: {mk.sum()} 首, 定数均值={y15[mk].mean():.2f}')
# miniburst_density 更好? (归一化时长)
mbd = np.array([o['feats'].get('miniburst_density', 0) for o in official])
mbd15 = np.array([o['feats'].get('miniburst_density', 0) for o in off15])
print(f'\nminiburst_density 官谱15+: P50={np.percentile(mbd15,50):.4f} P75={np.percentile(mbd15,75):.4f} P90={np.percentile(mbd15,90):.4f} P95={np.percentile(mbd15,95):.4f}')
print(f'Melodiniq density=0.017')
mk2 = mbd15 >= 0.017
print(f'15+官谱 density>=0.017: {mk2.sum()} 首, 定数均值={y15[mk2].mean():.2f}')
# 与定数相关性
from scipy.stats import spearmanr
print(f'\nminiburst_count rho(15+): {spearmanr(mb15, y15).statistic:.3f}')
print(f'miniburst_density rho(15+): {spearmanr(mbd15, y15).statistic:.3f}')
# 16.5+ 官谱的 miniburst
off165 = [o for o in official if o['diff'] >= 16.5]
mb165 = np.array([o['feats'].get('miniburst_count', 0) for o in off165])
print(f'\n官谱16.5+ ({len(off165)}首): miniburst P50={np.median(mb165):.0f} P25={np.percentile(mb165,25):.0f}')
print('DONE')