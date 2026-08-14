# -*- coding: utf-8 -*-
"""实验6c: 未上架 17+/18+ 段细节 — 段内Spearman + 多指/双指分布
"""
import os, sys, csv, numpy as np, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from scipy.stats import spearmanr

rows = []
with open(os.path.join(_ROOT, 'data', 'phira', 'unranked_4star_list.csv'), encoding='utf-8-sig') as f:
    rdr = csv.DictReader(f)
    for r_ in rdr:
        try:
            d = float(r_['difficulty']) if r_['difficulty'] else None
            if d is None or not (5.0 < d < 30.0): continue
            rows.append({'name': r_['name'], 'level': r_['level'], 'diff': d,
                         'rating': float(r_['rating']), 'rc': int(r_['ratingCount']),
                         'pred': float(r_['pred']), 'mf3': float(r_['mf3']), 'mf4': float(r_['mf4'])})
        except Exception:
            pass
ds = np.array([r['diff'] for r in rows]); ps = np.array([r['pred'] for r in rows])
mf = np.array([r['mf3'] for r in rows])

print('===== 17+ 段细节 =====')
for lo, hi, tag in [(17,17.5,'17-17.5'),(17.5,18,'17.5-18'),(18,18.5,'18-18.5'),(18.5,99,'18.5+')]:
    mk = np.where((ds >= lo) & (ds < hi))[0]
    if len(mk) < 8: continue
    rho, _ = spearmanr(ds[mk], ps[mk])
    g_mf = mk[mf[mk] >= 30]; g_df = mk[mf[mk] <= 5]
    print(f'  [{tag}]: n={len(mk)} Spearman={rho:.3f} bias={(ps[mk]-ds[mk]).mean():+.3f} MAE={np.abs(ps[mk]-ds[mk]).mean():.3f}')
    if len(g_mf) >= 5: print(f'      多指: n={len(g_mf)} bias={(ps[g_mf]-ds[g_mf]).mean():+.3f}')
    if len(g_df) >= 5: print(f'      双指: n={len(g_df)} bias={(ps[g_df]-ds[g_df]).mean():+.3f}')

# 18+ 预测分布: 社区18+的谱, 模型给了多少?
mk18 = np.where(ds >= 18)[0]
print(f'\n社区18+ (n={len(mk18)}) 的模型预测分布:')
for lo, hi, tag in [(15,16,'15-16'),(16,17,'16-17'),(17,17.5,'17-17.5'),(17.5,18,'17.5-18'),(18,99,'18+')]:
    g = mk18[(ps[mk18] >= lo) & (ps[mk18] < hi)]
    print(f'  预测[{tag}]: {len(g)}')

# 模型预测 17.5+ 的谱 社区定数分布
mkp = np.where(ps >= 17.5)[0]
print(f'\n模型预测17.5+ (n={len(mkp)}) 的社区定数分布:')
for lo, hi, tag in [(14,16,'14-16'),(16,17,'16-17'),(17,18,'17-18'),(18,99,'18+')]:
    g = mkp[(ds[mkp] >= lo) & (ds[mkp] < hi)]
    print(f'  社区定数[{tag}]: {len(g)}')
# 名字样例
idx = np.argsort(-ps[mkp])[:10]
print('\n模型预测最高10张:')
for i in idx:
    print(f'  {rows[mkp[i]]["name"][:30]:<32} 社区={rows[mkp[i]]["diff"]:.1f} 预测={rows[mkp[i]]["pred"]:.2f} mf3={rows[mkp[i]]["mf3"]:.0f}')
print('DONE')
