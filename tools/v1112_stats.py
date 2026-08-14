# -*- coding: utf-8 -*-
"""v11.12 完整统计: 整体/分段/官谱vs自制/误差分布"""
import os, sys, io, pickle, numpy as np, csv, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
from scipy.stats import spearmanr

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])

# CSV 预测 (生产值)
with open(os.path.join(_ROOT, 'data', 'phira', 'v1112_ranked_predictions.csv'), encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
pred_by_id = {int(x['id']): float(x['pred']) for x in rows}
ps = np.array([pred_by_id.get(r['id'], np.nan) for r in ranked])
ok = ~np.isnan(ps)
ps, ds2, ranked2 = ps[ok], ds[ok], [r for i, r in enumerate(ranked) if ok[i]]
errs = ps - ds2

print('='*70)
print(f'v11.12 整体 (n={len(ps)})')
print(f'  MAE={np.abs(errs).mean():.3f}  RMSE={np.sqrt((errs**2).mean()):.3f}  bias={errs.mean():+.3f}')
print(f'  rho={spearmanr(ps, ds2).statistic:.3f}')
print()
print('=== 分段统计 (真实diff段) ===')
for lo, hi, t2 in [(11,12,'11-12'),(12,13,'12-13'),(13,14,'13-14'),(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
    mk = np.where((ds2 >= lo) & (ds2 < hi))[0]
    if len(mk):
        e = errs[mk]
        print(f'  {t2:<8} n={len(mk):>3}  bias={e.mean():+.3f}  MAE={np.abs(e).mean():.3f}')
print()
print('=== 预测值段分布 (校准后) ===')
for lo, hi, t2 in [(11,13,'11-13'),(13,14,'13-14'),(14,15,'14-15'),(15,16,'15-16'),(16,17,'16-17'),(17,18,'17-18'),(18,99,'>=18')]:
    mk = np.where((ps >= lo) & (ps < hi))[0]
    if len(mk):
        print(f'  预测{t2:<8} n={len(mk):>3}  真实均值={ds2[mk].mean():.2f}  预测均值={ps[mk].mean():.2f}')
print()
print('=== 官谱 vs 上架 ===')
# 官谱: kyou_tags 310首
with open(os.path.join(_ROOT, 'data', 'phira', 'kyou_tags.json'), encoding='utf-8') as f:
    kt = json.load(f)
import re
def norm(s):
    return re.sub(r'[^0-9a-z一-鿿]', '', (s or '').lower())
kyou_names = [norm(item['song']) for item in kt]
off_mask = np.array([any(norm(r['name']) == kk or kk in norm(r['name']) or norm(r['name']) in kk for kk in kyou_names) for r in ranked2])
print(f'  官谱: n={off_mask.sum()}  MAE={np.abs(errs[off_mask]).mean():.3f}  bias={errs[off_mask].mean():+.3f}')
print(f'  非官谱: n={(~off_mask).sum()}  MAE={np.abs(errs[~off_mask]).mean():.3f}  bias={errs[~off_mask].mean():+.3f}')
print()
print('=== 误差绝对值最大 12 首 ===')
idx = np.argsort(-np.abs(errs))[:12]
for i in idx:
    print(f'  {ranked2[i]["name"][:26]:<28} diff={ds2[i]:.2f} 预测={ps[i]:.2f} err={errs[i]:+.2f}')
print()
print('=== 高估 Top8 / 低估 Top8 ===')
hi = np.argsort(-errs)[:8]
lo = np.argsort(errs)[:8]
print('高估:')
for i in hi:
    print(f'  {ranked2[i]["name"][:26]:<28} diff={ds2[i]:.2f} 预测={ps[i]:.2f} err={errs[i]:+.2f}')
print('低估:')
for i in lo:
    print(f'  {ranked2[i]["name"][:26]:<28} diff={ds2[i]:.2f} 预测={ps[i]:.2f} err={errs[i]:+.2f}')
print('DONE')