# -*- coding: utf-8 -*-
"""统计口径v2: 舍弃整数定数(diff为整数) + diff四舍五入到1位
用法: python tools/exp_v1110_stats.py [csv]
"""
import os, sys, csv, io
import numpy as np
from scipy.stats import spearmanr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, 'data', 'phira', 'v119_ranked_predictions.csv')
rows = []
with open(csv_path, encoding='utf-8-sig') as f:
    rd = csv.DictReader(f)
    for r_ in rd:
        try:
            d = round(float(r_['diff']), 1)  # 浮点清理 18.500002 → 18.5
            rows.append({'diff': d, 'pred': float(r_['pred']), 'err': float(r_['pred']) - d})
        except Exception:
            pass
all_n = len(rows)
# 舍弃整数定数 (早期phira只标大定数)
noint = [r for r in rows if r['diff'] != round(r['diff'])]
print(f'总计: {all_n} | 整数定数舍弃: {all_n - len(noint)} | 保留: {len(noint)}')
ds = np.array([r['diff'] for r in noint]); ps = np.array([r['pred'] for r in noint]); errs = ps - ds
print(f'\n===== 统计口径v2 (舍弃整数定数) =====')
print(f'MAE={np.abs(errs).mean():.3f} bias={errs.mean():+.3f} rho={spearmanr(ds, ps)[0]:.3f} RMSE={np.sqrt((errs**2).mean()):.3f}')
for lo, hi, tag in [(14,15,'14-15'),(15,16,'15-16'),(16,17,'16-17'),(17,99,'>=17')]:
    mk = np.where((ds >= lo) & (ds < hi))[0]
    if len(mk): print(f'  [{tag}]: n={len(mk)} bias={errs[mk].mean():+.3f} MAE={np.abs(errs[mk]).mean():.3f}')
print('DONE')
