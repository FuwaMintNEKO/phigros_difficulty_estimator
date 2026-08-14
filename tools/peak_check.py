# -*- coding: utf-8 -*-
"""尾杀/局部峰值特征验证: 最密10秒 密度x位移 在官谱中的分布"""
import os, sys, io, pickle, numpy as np, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes, time_to_seconds
from scipy.stats import spearmanr

def peak10(feats, cd):
    """最密10秒 密度x位移 (用已有特征近似: above_avg_density_mean x movement)"""
    dens = feats.get('above_avg_density_mean', 0)
    mov = feats.get('movement_per_second', 0)
    return dens * mov

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = cache['official']
off15 = [o for o in official if o['diff'] >= 15]
y15 = np.array([o['diff'] for o in off15])
dm = np.array([o['feats'].get('above_avg_density_mean',0) * o['feats'].get('movement_per_second',0) for o in off15])
print(f'dens×mov 官谱15+: rho={spearmanr(dm, y15).statistic:.3f}')
mk = dm >= 400
print(f'  >=400: {mk.sum()} 首, 定数均值={y15[mk].mean():.2f}')
mk2 = dm >= 300
print(f'  >=300: {mk2.sum()} 首, 定数均值={y15[mk2].mean():.2f}')
# Melodiniq
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
from feature_extractor import extract_features
fm = extract_features(cd, speed=1.0)
print(f'\nMelodiniq dens×mov = {fm.get("above_avg_density_mean",0)*fm.get("movement_per_second",0):.1f}')
print(f'  百分位: {np.mean(dm <= fm.get("above_avg_density_mean",0)*fm.get("movement_per_second",0))*100:.0f}%')
# 与运动密度指数对比
mdi = np.array([o['feats'].get('movement_density_index', 0) for o in off15])
print(f'movement_density_index 官谱15+: rho={spearmanr(mdi, y15).statistic:.3f} P50={np.median(mdi):.0f}')
print(f'Melodiniq mdi=130.09 → {np.mean(mdi <= 130.09)*100:.0f}%')
print('DONE')