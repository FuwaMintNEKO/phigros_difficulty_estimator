# -*- coding: utf-8 -*-
"""ranked 清单表现报告 (v114_ranked_predictions.csv)
"""
import os, sys, csv, numpy as np, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from scipy.stats import spearmanr

rows = []
with open(os.path.join(_ROOT, 'data', 'phira', 'v114_ranked_predictions.csv'), encoding='utf-8-sig') as f:
    rdr = csv.DictReader(f)
    for r_ in rdr:
        try:
            rows.append({'id': int(r_['id']), 'name': r_['name'], 'level': r_['level'],
                         'diff': float(r_['diff']), 'pred': float(r_['pred']), 'err': float(r_['err']),
                         'gb': float(r_['gb']), 'boost': float(r_['boost']),
                         'mf3': float(r_['mf3']), 'dens': float(r_['dens']), 'notes': int(r_['notes'])})
        except Exception:
            pass
print(f'ranked 清单: {len(rows)} 张')
ds = np.array([r['diff'] for r in rows]); ps = np.array([r['pred'] for r in rows]); errs = ps - ds
rho, _ = spearmanr(ds, ps)
print(f'\n===== 总体 =====')
print(f'MAE={np.abs(errs).mean():.3f}  bias={errs.mean():+.3f}  Spearman={rho:.3f}')

print('\n===== 按社区定数分段 =====')
for lo, hi, tag in [(0,14,'<14'),(14,15,'14-15'),(15,16,'15-16'),(16,17,'16-17'),(17,99,'>=17')]:
    mk = np.where((ds >= lo) & (ds < hi))[0]
    if len(mk):
        r2, _ = spearmanr(ds[mk], ps[mk])
        print(f'  [{tag}]: n={len(mk):3d}  bias={errs[mk].mean():+.3f}  MAE={np.abs(errs[mk]).mean():.3f}  段内Spearman={r2:.3f}')

print('\n===== 多指/双指/混合 (16+段) =====')
for lo in [14.0, 16.0]:
    mk = np.where(ds >= lo)[0]
    if len(mk) < 5: continue
    print(f'  >= {lo} (n={len(mk)}):')
    for lbl, cond in [('多指(mf3>=30)', [r['mf3'] for r in rows]), ]:
        pass
    mf = np.array([r['mf3'] for r in rows])
    for lbl, cond in [('多指', mf >= 30), ('双指', mf <= 5), ('混合', (mf > 5) & (mf < 30))]:
        g = mk[cond[mk]]
        if len(g):
            print(f'    {lbl}: n={len(g):3d}  bias={errs[g].mean():+.3f}  MAE={np.abs(errs[g]).mean():.3f}')

print('\n===== 最离谱的 10 张 (|err| 最大) =====')
idx = np.argsort(-np.abs(errs))[:10]
for i in idx:
    print(f'  {rows[i]["name"][:34]:<36} {rows[i]["level"][:12]:<12} 社区={ds[i]:.1f} 预测={ps[i]:.2f} err={errs[i]:+.2f} mf3={rows[i]["mf3"]:.0f}')

print('\n===== 预测最高 12 张 =====')
idx = np.argsort(-ps)[:12]
for i in idx:
    print(f'  {rows[i]["name"][:34]:<36} {rows[i]["level"][:12]:<12} 社区={ds[i]:.1f} 预测={ps[i]:.2f} err={errs[i]:+.2f} mf3={rows[i]["mf3"]:.0f}')

print('\n===== 社区最高 12 张 =====')
idx = np.argsort(-ds)[:12]
for i in idx:
    print(f'  {rows[i]["name"][:34]:<36} {rows[i]["level"][:12]:<12} 社区={ds[i]:.1f} 预测={ps[i]:.2f} err={errs[i]:+.2f} mf3={rows[i]["mf3"]:.0f}')

# 等级分布
print('\n===== 等级分布 =====')
from collections import Counter
c = Counter(r['level'].upper() for r in rows)
for k, v in c.most_common():
    print(f'  {k}: {v}')
print('DONE')
