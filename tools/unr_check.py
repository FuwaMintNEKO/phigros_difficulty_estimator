# -*- coding: utf-8 -*-
"""验证unranked预测结果: 分布检查"""
import os, sys, io, csv, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'unranked_4star_list.csv')
with open(p, encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
print('行数:', len(rows))
preds = np.array([float(r['pred']) for r in rows if r.get('pred')])
print(f'预测分布: P10={np.percentile(preds,10):.1f} P50={np.percentile(preds,50):.1f} P90={np.percentile(preds,90):.1f} max={preds.max():.1f}')
print(f'  >=15: {np.sum(preds>=15)}, >=16: {np.sum(preds>=16)}, >=16.5: {np.sum(preds>=16.5)}, >=17: {np.sum(preds>=17)}')
# 预测最高20首
idx = np.argsort(-preds)[:20]
print('\n预测最高20首:')
for i in idx:
    r = rows[i]
    print(f"  {r['name'][:24]:<26} pred={float(r['pred']):.2f} gb={float(r['gb']):.2f} boost={float(r['boost']):.2f} diff字段={r['difficulty']} rating={r.get('rating','')}")
# 与difficulty字段对比
diffs = []
for r in rows:
    try:
        d = float(r['difficulty'])
        if 5 <= d <= 20: diffs.append((float(r['pred']), d))
    except: pass
if diffs:
    arr = np.array(diffs)
    print(f'\n与谱师自标diff对比 (n={len(arr)}):')
    print(f'  预测-自标 bias={np.mean(arr[:,0]-arr[:,1]):+.3f} MAE={np.mean(np.abs(arr[:,0]-arr[:,1])):.3f}')
print('DONE')