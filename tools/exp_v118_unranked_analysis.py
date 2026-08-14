# -*- coding: utf-8 -*-
"""unranked 5894 详细分析 (v11.7b CSV + tags)
"""
import os, csv, io, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = r'D:\Trae项目\新建文件夹\phigros_difficulty_estimator'
tags = {}
with open(os.path.join(_ROOT, 'data', 'phira', 'unranked_tags.csv'), encoding='utf-8-sig') as f:
    rd = csv.DictReader(f)
    for r_ in rd:
        tags[int(r_['id'])] = r_['tags']
rows = []
with open(os.path.join(_ROOT, 'data', 'phira', 'unranked_4star_list.csv'), encoding='utf-8-sig') as f:
    rd = csv.DictReader(f)
    for r_ in rd:
        try:
            cid = int(r_['id'])
            d = float(r_['difficulty']) if r_['difficulty'] else None
            if d is None or not (5.0 < d < 25.0): continue
            rows.append({'name': r_['name'], 'level': r_['level'], 'diff': d, 'pred': float(r_['pred']),
                         'gb': float(r_['gb']), 'boost': float(r_['boost']), 'rating': float(r_['rating']),
                         'rc': int(r_['ratingCount']), 'mf3': float(r_['mf3']), 'tags': tags.get(cid, '-')})
        except Exception:
            pass
print(f'===== UNRANKED {len(rows)} 张 (清洗后 5<d<25) =====')
ds = np.array([r['diff'] for r in rows]); ps = np.array([r['pred'] for r in rows]); errs = ps - ds
print(f'整体: MAE={np.abs(errs).mean():.3f} bias={errs.mean():+.3f} rho={spearmanr(ds, ps)[0]:.3f} RMSE={np.sqrt((errs**2).mean()):.3f}')
print('\n按社区定数段:')
for lo, hi, tag in [(5,14,'<14'),(14,15,'14-15'),(15,16,'15-16'),(16,17,'16-17'),(17,18,'17-18'),(18,99,'>=18')]:
    mk = np.where((ds >= lo) & (ds < hi))[0]
    if len(mk): print(f'  [{tag}]: n={len(mk):4d} bias={errs[mk].mean():+.3f} MAE={np.abs(errs[mk]).mean():.3f}')
# 高评分过滤
print('\n高评分 (rt>=0.9, rc>=100):')
hi_rows = [r for r in rows if r['rating'] >= 0.9 and r['rc'] >= 100]
if hi_rows:
    ds2 = np.array([r['diff'] for r in hi_rows]); ps2 = np.array([r['pred'] for r in hi_rows]); er2 = ps2 - ds2
    print(f'  n={len(hi_rows)} MAE={np.abs(er2).mean():.3f} bias={er2.mean():+.3f} rho={spearmanr(ds2, ps2)[0]:.3f}')
    for lo, hi, tag in [(14,16,'14-16'),(16,17,'16-17'),(17,18,'17-18'),(18,99,'>=18')]:
        mk = np.where((ds2 >= lo) & (ds2 < hi))[0]
        if len(mk): print(f'    [{tag}]: n={len(mk)} bias={er2[mk].mean():+.3f}')
# 标签组 (高评分 16+)
print('\n标签组偏差 (rt>=0.9, rc>=100, 16+):')
sel = [r for r in hi_rows if r['diff'] >= 16]
g = defaultdict(list)
for r in sel:
    if r['tags'] == '-': g['(无标签)'].append(r['pred'] - r['diff'])
    else:
        for t in r['tags'].split('+'): g[t].append(r['pred'] - r['diff'])
for t, es in sorted(g.items(), key=lambda x: -abs(sum(x[1])/len(x[1]))):
    if len(es) >= 15: print(f'  {t:<10} n={len(es):>4} bias={sum(es)/len(es):+.3f}')
# 大误差案例 (高评分)
print('\n高评分大误差案例 (|err|>=1.5):')
big = [r for r in hi_rows if abs(r['pred'] - r['diff']) >= 1.5]
big.sort(key=lambda x: abs(x['pred'] - x['diff']), reverse=True)
for r in big[:20]:
    print(f'  {r["pred"]-r["diff"]:+.2f} {r["name"][:22]:<24} 社区={r["diff"]:.1f} 预测={r["pred"]:.2f} rt={r["rating"]:.2f} [{r["tags"]}]')
print('DONE')
