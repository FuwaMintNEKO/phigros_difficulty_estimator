# -*- coding: utf-8 -*-
"""18.5+ 双指谱审查 + SP谱排除后的整体统计
"""
import os, sys, csv, numpy as np, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from scipy.stats import spearmanr

meta = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'unranked_all.json'), encoding='utf-8'))
meta_by_id = {m['id']: m for m in meta}

rows = []
with open(os.path.join(_ROOT, 'data', 'phira', 'unranked_4star_list.csv'), encoding='utf-8-sig') as f:
    rdr = csv.DictReader(f)
    for r_ in rdr:
        try:
            d = float(r_['difficulty']) if r_['difficulty'] else None
            if d is None or not (5.0 < d < 30.0): continue
            m = meta_by_id.get(int(r_['id']), {})
            lv = (m.get('level') or '').upper()
            rows.append({'id': int(r_['id']), 'name': r_['name'], 'level': lv, 'diff': d,
                         'pred': float(r_['pred']), 'mf3': float(r_['mf3'])})
        except Exception:
            pass

# SP谱统计
sp = [r for r in rows if 'SP' in r['level']]
print(f'SP谱: {len(sp)} 张 (排除出趋势评估)')
rows_nosp = [r for r in rows if 'SP' not in r['level']]
ds = np.array([r['diff'] for r in rows_nosp]); ps = np.array([r['pred'] for r in rows_nosp])
mf = np.array([r['mf3'] for r in rows_nosp])
rho, _ = spearmanr(ds, ps)
print(f'排除SP后: n={len(rows_nosp)} Spearman={rho:.3f} bias={(ps-ds).mean():+.3f}')

# 18.5+ 双指谱 (社区标18.5+ 但 mf3<=5)
mk = np.where((ds >= 18.5) & (mf <= 5))[0]
print(f'\n社区18.5+ 双指谱 (n={len(mk)}):')
for i in mk:
    print(f'  {rows_nosp[i]["name"][:32]:<34} 社区={ds[i]:.1f} 预测={ps[i]:.2f} mf3={mf[i]:.0f} level={rows_nosp[i]["level"][:12]}')

# 17-18 段 (排除SP) 细分
print('\n17-18段 (排除SP, n>=10):')
for lo, hi, tag in [(17,17.5,'17-17.5'),(17.5,18,'17.5-18')]:
    mk = np.where((ds >= lo) & (ds < hi))[0]
    rho2, _ = spearmanr(ds[mk], ps[mk])
    print(f'  [{tag}]: n={len(mk)} Spearman={rho2:.3f} bias={(ps[mk]-ds[mk]).mean():+.3f}')
    for lbl, cond in [('多指', mf[mk]>=30), ('双指', mf[mk]<=5), ('混合', (mf[mk]>5)&(mf[mk]<30))]:
        g = mk[cond]
        if len(g) >= 5: print(f'      {lbl}: n={len(g)} bias={(ps[g]-ds[g]).mean():+.3f}')
print('DONE')
