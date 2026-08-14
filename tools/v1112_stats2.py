# -*- coding: utf-8 -*-
"""修正官谱匹配 + 17+段详细统计"""
import os, sys, io, pickle, numpy as np, csv, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])
with open(os.path.join(_ROOT, 'data', 'phira', 'v1112_ranked_predictions.csv'), encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
pred_by_id = {int(x['id']): float(x['pred']) for x in rows}
ps = np.array([pred_by_id.get(r['id'], np.nan) for r in ranked])
errs = ps - ds

def norm(s):
    return re.sub(r'[^0-9a-z一-鿿]', '', (s or '').lower())
with open(os.path.join(_ROOT, 'data', 'phira', 'kyou_tags.json'), encoding='utf-8') as f:
    kt = json.load(f)
kyou_names = [norm(item['song']) for item in kt]
kyou_names = [k for k in kyou_names if k]
print('kyou 官谱名称数:', len(kyou_names))
off_mask = np.array([any(norm(r['name']) == kk for kk in kyou_names) for r in ranked])
print(f'官谱匹配: {off_mask.sum()}/{len(ranked)}')
print(f'官谱: n={off_mask.sum()} MAE={np.abs(errs[off_mask]).mean():.3f} bias={errs[off_mask].mean():+.3f}')
print(f'上架自制: n={(~off_mask).sum()} MAE={np.abs(errs[~off_mask]).mean():.3f} bias={errs[~off_mask].mean():+.3f}')
print()
print('=== 官谱 16.5+ 段 ===')
for lo, hi, t2 in [(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
    mk = np.where((ds >= lo) & (ds < hi) & off_mask)[0]
    if len(mk):
        print(f'  {t2}: n={len(mk)} bias={errs[mk].mean():+.3f} MAE={np.abs(errs[mk]).mean():.3f}')
print()
print('=== 上架自制 16.5+ 段 ===')
for lo, hi, t2 in [(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
    mk = np.where((ds >= lo) & (ds < hi) & ~off_mask)[0]
    if len(mk):
        print(f'  {t2}: n={len(mk)} bias={errs[mk].mean():+.3f} MAE={np.abs(errs[mk].mean()):.3f}')
print()
print('=== 17+ 全部谱 (真实diff>=17) ===')
for i in np.where(ds >= 17)[0]:
    print(f'  {ranked[i]["name"][:28]:<30} diff={ds[i]:.2f} 预测={ps[i]:.2f} err={errs[i]:+.2f} 官谱={"Y" if off_mask[i] else "N"}')
print()
print('=== 预测>=17 的谱 (模型认为17+) ===')
for i in np.where(ps >= 17)[0]:
    print(f'  {ranked[i]["name"][:28]:<30} diff={ds[i]:.2f} 预测={ps[i]:.2f} err={errs[i]:+.2f} 官谱={"Y" if off_mask[i] else "N"}')
print('DONE')