# -*- coding: utf-8 -*-
"""micro_max_0.0625beat 深入: 官谱分布 + Melodiniq定位 + 语义验证"""
import os, sys, io, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from scipy.stats import spearmanr
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
import app as app_mod

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = cache['official']
off15 = [o for o in official if o['diff'] >= 15]
y15 = np.array([o['diff'] for o in off15])
mm = np.array([o['feats'].get('micro_max_0.0625beat', 0) for o in off15])
print(f'micro_max_0.0625beat 官谱15+: 值分布={np.unique(mm)[:10]}')
print(f'  P50={np.percentile(mm,50)} P75={np.percentile(mm,75)} P90={np.percentile(mm,90)} P95={np.percentile(mm,95)}')
mk = mm >= 4
print(f'  >=4: {mk.sum()} 首, 定数均值={y15[mk].mean():.2f}')
mk2 = mm >= 6
print(f'  >=6: {mk2.sum()} 首, 定数均值={y15[mk2].mean():.2f}')
print(f'  rho(15+): {spearmanr(mm, y15).statistic:.3f}')
# micro_max_0.125beat (1/8拍窗)
mm2 = np.array([o['feats'].get('micro_max_0.125beat', 0) for o in off15])
mk3 = mm2 >= 4
print(f'\nmicro_max_0.125beat >=4: {mk3.sum()} 首, 定数均值={y15[mk3].mean():.2f}')
print(f'  0.125 rho: {spearmanr(mm2, y15).statistic:.3f}')
# 16.5+ 官谱的 micro_max
off165 = [o for o in official if o['diff'] >= 16.5]
for k in ['micro_max_0.0625beat','micro_max_0.125beat','micro_max_0.25beat']:
    v = np.array([o['feats'].get(k, 0) for o in off165])
    print(f'  16.5+ {k}: P50={np.median(v):.0f} P90={np.percentile(v,90):.0f}')
# Melodiniq/Verrückt
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
fm = extract_features(cd, speed=1.0)
p2 = os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')
with open(p2, 'rb') as f:
    cd2, _ = load_chart_from_bytes(f.read())
fv = extract_features(cd2, speed=1.0)
for k in ['micro_max_0.0625beat','micro_max_0.125beat','micro_max_0.25beat','core_micro_max_0.0625beat']:
    print(f'  {k}: Melodiniq={fm.get(k,0)} Verrückt={fv.get(k,0)}')
print('DONE')