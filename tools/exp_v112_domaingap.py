# -*- coding: utf-8 -*-
"""官谱 vs 上架谱 特征域偏移扫描 (IN/AT段对齐, Cohen's d)
找出自制谱系统性偏移的特征 → 偏差来源
"""
import os, sys, pickle, numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)

off = cache['official']
rkd = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]

def cohens_d(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*a.var() + (nb-1)*b.var()) / (na+nb-2))
    return (a.mean() - b.mean()) / max(sp, 1e-9)

# 对齐段: 官谱 IN/AT (11-17.6) vs 上架谱 (11-17.6)
off_hi = [f for f in off if 11 <= f['diff'] <= 17.6]
rkd_hi = [r for r in rkd if 11 <= r['diff'] <= 17.6]
print(f'官谱 11-17.6: {len(off_hi)} | 上架 11-17.6: {len(rkd_hi)}')

# 所有特征
all_keys = sorted(off_hi[0]['feats'].keys())
results = []
for k in all_keys:
    a = [f['feats'].get(k, 0) for f in off_hi]
    b = [r['feats'].get(k, 0) for r in rkd_hi]
    if max(np.max(a), np.max(b)) < 1e-6: continue
    d = cohens_d(b, a)  # 正 = 上架高于官谱
    results.append((k, d, np.mean(a), np.mean(b)))

results.sort(key=lambda x: -abs(x[1]))
print('\n=== 上架谱相对官谱偏移最大 Top 40 (Cohen\'s d, 正=上架偏高) ===')
for k, d, ma, mb in results[:40]:
    print(f'  {k:<34} d={d:+.2f} 官={ma:>10.2f} 上={mb:>10.2f}')
print('\n=== 偏移最小(接近对齐) Top 15 ===')
for k, d, ma, mb in results[-15:]:
    print(f'  {k:<34} d={d:+.2f} 官={ma:>10.2f} 上={mb:>10.2f}')
