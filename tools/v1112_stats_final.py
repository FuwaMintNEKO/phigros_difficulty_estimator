# -*- coding: utf-8 -*-
"""过滤后统计: 上架区(去特殊) + 去整数标级"""
import os, sys, io, json, pickle, numpy as np, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from scipy.stats import spearmanr

charts = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
up_ids = {c['id'] for c in charts['上架']}
sp_ids = {c['id'] for c in charts['特殊']}
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10 and r['id'] in up_ids]
ds = np.array([round(r['diff'],1) for r in ranked])
with open(os.path.join(_ROOT, 'data', 'phira', 'v1112_ranked_predictions.csv'), encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
pred_by_id = {int(x['id']): float(x['pred']) for x in rows}
ps = np.array([pred_by_id.get(r['id'], np.nan) for r in ranked])
ok = ~np.isnan(ps)
ps, ds2, ranked2 = ps[ok], ds[ok], [r for i, r in enumerate(ranked) if ok[i]]
errs = ps - ds2

# 特殊区谱在剩余中的(不应有)
sp_in = [r['id'] for r in ranked2 if r['id'] in sp_ids]
print(f'上架区谱: {len(ranked2)} (应无特殊谱: {len(sp_in)})')
# 整数标级
int_mask = np.abs(ds2 - np.round(ds2)) < 1e-6
print(f'其中整数标级: {int_mask.sum()}, 保留非整数: {(~int_mask).sum()}')
ps_f, ds_f, ranked_f, errs_f = ps[~int_mask], ds2[~int_mask], [r for i, r in enumerate(ranked2) if not int_mask[i]], errs[~int_mask]

print('\n' + '='*70)
print(f'v11.12 上架(去特殊+去整数) n={len(ps_f)}')
print(f'  MAE={np.abs(errs_f).mean():.3f}  RMSE={np.sqrt((errs_f**2).mean()):.3f}  bias={errs_f.mean():+.3f}  rho={spearmanr(ps_f, ds_f).statistic:.3f}')
print()
print('=== 分段 ===')
for lo, hi, t2 in [(12,13,'12-13'),(13,14,'13-14'),(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
    mk = np.where((ds_f >= lo) & (ds_f < hi))[0]
    if len(mk):
        e = errs_f[mk]
        print(f'  {t2:<9} n={len(mk):>3}  bias={e.mean():+.3f}  MAE={np.abs(e).mean():.3f}')
print()
print('=== 误差最大 Top10 ===')
idx = np.argsort(-np.abs(errs_f))[:10]
for i in idx:
    print(f'  {ranked_f[i]["name"][:28]:<30} diff={ds_f[i]:.2f} 预测={ps_f[i]:.2f} err={errs_f[i]:+.2f}')
print()
print('=== 高估 Top6 / 低估 Top6 ===')
hi = np.argsort(-errs_f)[:6]
lo = np.argsort(errs_f)[:6]
print('高估:')
for i in hi:
    print(f'  {ranked_f[i]["name"][:28]:<30} diff={ds_f[i]:.2f} 预测={ps_f[i]:.2f} err={errs_f[i]:+.2f}')
print('低估:')
for i in lo:
    print(f'  {ranked_f[i]["name"][:28]:<30} diff={ds_f[i]:.2f} 预测={ps_f[i]:.2f} err={errs_f[i]:+.2f}')
print('DONE')