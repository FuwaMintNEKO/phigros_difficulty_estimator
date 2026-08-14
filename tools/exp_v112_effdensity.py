# -*- coding: utf-8 -*-
"""1. AT段密度gap检查 2. eff化密度与定数相关性 (官谱)"""
import os, sys, pickle, numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)

off = cache['official']
rkd = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]

# 1. 分段密度 gap
print('=== 密度 gap by 段 (上架 - 官谱) ===')
for tag, lo, hi, off_filter, rkd_filter in [
    ('IN段 11-14', 11, 14, lambda f: f['diff'] >= 11, lambda r: r['diff'] >= 11),
    ('IN段 14-16', 14, 16, lambda f: f['diff'] >= 14, lambda r: r['diff'] >= 14),
    ('IN段 16-16.5', 16, 16.5, lambda f: f['diff'] >= 16, lambda r: r['diff'] >= 16),
    ('AT段 16.5+', 16.5, 99, lambda f: f['diff'] >= 16.5, lambda r: r['diff'] >= 16.5),
]:
    pass
# 简化: 按官谱/上架定数段直接比
for lo, hi, tag in [(11, 14, '11-14'), (14, 16, '14-16'), (16, 16.5, '16-16.5'), (16.5, 99, '16.5+')]:
    o = [f['feats'].get('above_avg_density_mean', 0) for f in off if lo <= f['diff'] < hi]
    r = [x['feats'].get('above_avg_density_mean', 0) for x in rkd if lo <= x['diff'] < hi]
    if o and r:
        print(f'  [{tag}]: 官={np.mean(o):.2f} (n={len(o)}) 上={np.mean(r):.2f} (n={len(r)}) gap={np.mean(r)-np.mean(o):+.2f}')

# 2. 官谱: eff化密度 相关性
print('\n=== 官谱特征与定数相关性 (IN/AT, 11-17.6) ===')
off_hi = [f for f in off if f['diff'] >= 11]
y = np.array([f['diff'] for f in off_hi])
for k in ['above_avg_density_mean', 'real_core_notes_per_second', 'eff_avg_tps_1s', 'eff_peak_tps_1s']:
    v = np.array([f['feats'].get(k, 0) for f in off_hi])
    print(f'  {k:<28} corr={np.corrcoef(v, y)[0,1]:+.3f}')
# eff化密度候选: eff_avg / dens (多押撑密度时低)
r_ea = np.array([f['feats'].get('eff_avg_tps_1s', 0) / max(f['feats'].get('above_avg_density_mean', 0), 0.1) for f in off_hi])
print(f'  eff_avg/dens (effratio)      corr={np.corrcoef(r_ea, y)[0,1]:+.3f}')
# eff_peak 与 dens 的组合
r_ep = np.array([f['feats'].get('eff_peak_tps_1s', 0) / max(f['feats'].get('above_avg_density_mean', 0), 0.1) for f in off_hi])
print(f'  eff_peak/dens                corr={np.corrcoef(r_ep, y)[0,1]:+.3f}')
# 乘积型: dens × effratio (去冗余后的密度强度)
r_mix = np.array([f['feats'].get('above_avg_density_mean', 0) * f['feats'].get('eff_avg_tps_1s', 0) / max(f['feats'].get('above_avg_density_mean', 0), 0.1) for f in off_hi])
print(f'  eff_avg (即上式, 重复)        corr={np.corrcoef(r_mix, y)[0,1]:+.3f}')

# 3. 官谱: eff_avg/dens 分布 vs 上架
print('\n=== effratio (eff_avg/dens) 分布 ===')
for tag, data in [('官谱11+', off_hi), ('上架11+', [r for r in rkd if r['diff'] >= 11])]:
    vals = [f['feats'].get('eff_avg_tps_1s', 0) / max(f['feats'].get('above_avg_density_mean', 0), 0.1) for f in data]
    print(f'  {tag}: mean={np.mean(vals):.2f} p25={np.percentile(vals,25):.2f} p50={np.percentile(vals,50):.2f} p75={np.percentile(vals,75):.2f}')
