# -*- coding: utf-8 -*-
"""unranked 元数据分析: rating/ratingCount 分布, 筛选1000张的可行性"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
meta = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'unranked_all.json'), encoding='utf-8'))
print('unranked_all 总数:', len(meta))
# 有 rating 的
has_r = [c for c in meta if c.get('rating') is not None and c.get('ratingCount', 0) > 0]
print('有rating且ratingCount>0:', len(has_r))
ratings = np.array([c['rating'] for c in has_r])
counts = np.array([c['ratingCount'] for c in has_r])
print(f'rating: P50={np.percentile(ratings,50):.3f} P75={np.percentile(ratings,75):.3f} P90={np.percentile(ratings,90):.3f}')
print(f'ratingCount: P50={np.percentile(counts,50):.0f} P75={np.percentile(counts,75):.0f} P90={np.percentile(counts,90):.0f} max={counts.max():.0f}')
# 组合筛选
for rth, cth in [(0.9, 50), (0.85, 50), (0.85, 30), (0.8, 20), (0.75, 10), (0.7, 5)]:
    sel = [c for c in has_r if c['rating'] >= rth and c['ratingCount'] >= cth]
    print(f'rating>={rth} & count>={cth}: {len(sel)} 张')
    # 定数分布(difficulty字段, 谱师自标)
    diffs = [c.get('difficulty') for c in sel if c.get('difficulty')]
    d = np.array([float(x) for x in diffs])
    print(f'   difficulty 分布: n={len(d)} P50={np.percentile(d,50):.1f} P90={np.percentile(d,90):.1f} max={d.max():.1f} (16.5+: {np.sum(d>=16.5)}, 17+: {np.sum(d>=17)})')
# 看看 top rating 谱
print('\ntop20 (rating*count 综合):')
def score(c): return c['rating'] * min(c['ratingCount'], 500)
top = sorted(has_r, key=score, reverse=True)[:20]
for c in top:
    print(f"  {c['name'][:26]:<28} rating={c['rating']:.3f} count={c['ratingCount']} diff={c.get('difficulty')} lv={c.get('level')}")
print('DONE')