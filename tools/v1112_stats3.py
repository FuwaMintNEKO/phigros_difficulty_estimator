# -*- coding: utf-8 -*-
"""官谱匹配修复: 包含匹配"""
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
kyou_names = [k for k in kyou_names if k and len(k) >= 2]

# 匹配策略: kyou名 是 ranked名 的子串 或 反向 (排除太短)
def match(nm):
    for kk in kyou_names:
        if kk in nm or nm in kk:
            return True
    return False
off_mask = np.array([match(norm(r['name'])) for r in ranked])
print(f'官谱匹配: {off_mask.sum()}/{len(ranked)}')
print(f'官谱: n={off_mask.sum()} MAE={np.abs(errs[off_mask]).mean():.3f} bias={errs[off_mask].mean():+.3f}')
print(f'上架自制: n={(~off_mask).sum()} MAE={np.abs(errs[~off_mask]).mean():.3f} bias={errs[~off_mask].mean():+.3f}')
# 官谱分段
print()
print('=== 官谱分段 ===')
for lo, hi, t2 in [(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
    mk = np.where((ds >= lo) & (ds < hi) & off_mask)[0]
    if len(mk):
        print(f'  {t2}: n={len(mk)} bias={errs[mk].mean():+.3f} MAE={np.abs(errs[mk]).mean():.3f}')
print()
print('=== 上架自制分段 ===')
for lo, hi, t2 in [(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
    mk = np.where((ds >= lo) & (ds < hi) & ~off_mask)[0]
    if len(mk):
        print(f'  {t2}: n={len(mk)} bias={errs[mk].mean():+.3f} MAE={np.abs(errs[mk]).mean():+.3f}')
print('DONE')