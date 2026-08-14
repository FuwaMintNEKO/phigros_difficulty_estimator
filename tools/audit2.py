# -*- coding: utf-8 -*-
"""彻查特征计算: 1) fast_ms_100 rho≈0 2) pattern_switch负相关 3) Melodiniq核心特征验证"""
import os, sys, io, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from scipy.stats import spearmanr
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = cache['official']
off15 = [o for o in official if o['diff'] >= 15]
y15 = np.array([o['diff'] for o in off15])

# 1) fast_ms_100_ratio: 为什么rho≈0? 看分布
f100 = np.array([o['feats'].get('fast_ms_100_ratio', 0) for o in off15])
f050 = np.array([o['feats'].get('fast_ms_050_ratio', 0) for o in off15])
print('fast_ms_100_ratio 分布: min={:.2f} P25={:.2f} P50={:.2f} P75={:.2f} max={:.2f}'.format(
    f100.min(), np.percentile(f100,25), np.percentile(f100,50), np.percentile(f100,75), f100.max()))
print('fast_ms_050_ratio 分布: min={:.2f} P25={:.2f} P50={:.2f} P75={:.2f} max={:.2f}'.format(
    f050.min(), np.percentile(f050,25), np.percentile(f050,50), np.percentile(f050,75), f050.max()))
# 高定数谱的 fast_ms
hi = y15 >= 16.5
print(f'16.5+: f100 P50={np.percentile(f100[hi],50):.3f} f050 P50={np.percentile(f050[hi],50):.3f}')
lo = y15 < 15.5
print(f'<15.5: f100 P50={np.percentile(f100[lo],50):.3f} f050 P50={np.percentile(f050[lo],50):.3f}')

# 2) pattern_switch_rate 负相关?
psr = np.array([o['feats'].get('pattern_switch_rate', 0) for o in off15])
print(f'\npattern_switch_rate: rho={spearmanr(psr, y15).statistic:.3f}')
print(f'  16.5+ P50={np.percentile(psr[hi],50):.3f} vs <15.5 P50={np.percentile(psr[lo],50):.3f}')

# 3) Melodiniq vs 官谱16.5+ 核心特征分布对比
print('\nMelodiniq vs 官谱16.5+:')
for k in ['miniburst_count','micro_max_0.0625beat','fast_ms_050_ratio','above_avg_density_mean','movement_per_second','real_core_notes_per_second','duration_sec']:
    v165 = np.array([o['feats'].get(k, 0) for o in official if o['diff'] >= 16.5])
    print(f'  {k:<32} 官谱16.5+ P50={np.median(v165):.2f} P75={np.percentile(v165,75):.2f} P90={np.percentile(v165,90):.2f}')
print('DONE')