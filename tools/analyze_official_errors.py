# -*- coding: utf-8 -*-
"""官谱误差分段分析：按定数区间统计预测偏差"""
import os, sys, json, pickle
import numpy as np

ROOT = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator'
sys.path.insert(0, ROOT)
from feature_extractor import extract_features
from unified_parser import load_chart_from_bytes

with open(os.path.join(ROOT, 'models', '6dim_model_v7.pkl'), 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']

import importlib.util
spec = importlib.util.spec_from_file_location('appmod', os.path.join(ROOT, 'app.py'))
appmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(appmod)

def predict(feats):
    if not feats:
        return None
    x = np.array([[feats.get(n, 0) for n in FN]])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    p_boost, dims, _ = appmod.compute_boost(feats)
    return p_gb + p_boost, p_gb, p_boost, dims['categories'], feats

from data_loader import load_difficulty_tsv, find_chart_files
diffs = load_difficulty_tsv(os.path.join(ROOT, 'data', 'info', 'difficulty.tsv'))
chart_files = find_chart_files(os.path.join(ROOT, 'data', 'chart'))

rows = []
for folder, info in chart_files.items():
    sid = info['song_id']
    if sid not in diffs:
        continue
    for lv, path in info['levels'].items():
        if lv not in diffs[sid]:
            continue
        try:
            with open(path, 'rb') as f:
                cd, pe = load_chart_from_bytes(f.read())
            feats = extract_features(cd)
            if not feats:
                continue
            r = predict(feats)
            real = diffs[sid][lv]
            rows.append((real, r[0], r[1], r[2], r[3], folder, lv, feats))
        except Exception:
            pass

rows.sort(key=lambda x: x[0])
errs = np.array([p - r for r, p, *_ in rows])
reals = np.array([r for r, *_ in rows])

print(f'总计 {len(rows)} 个谱面')
print(f'整体: MAE={np.mean(np.abs(errs)):.3f}  Bias={np.mean(errs):+.3f}  median={np.median(errs):+.3f}')

# 按定数分段
bins = [(4,7),(7,9),(9,11),(11,13),(13,14.5),(14.5,16),(16,17.5),(17.5,20)]
for lo, hi in bins:
    mask = (reals >= lo) & (reals < hi)
    if mask.sum() == 0:
        continue
    e = errs[mask]
    print(f'  定数[{lo:4.1f},{hi:4.1f}): n={mask.sum():4d}  Bias={np.mean(e):+.3f}  MAE={np.mean(np.abs(e)):.3f}  '
          f'GBavg={np.mean([rows[i][2] for i in np.where(mask)[0]]):.2f}  Boostavg={np.mean([rows[i][3] for i in np.where(mask)[0]]):.2f}')

print()
print('最离谱的 10 个（按 |误差| 排序）:')
worst = sorted(rows, key=lambda r: -abs(r[1]-r[0]))[:10]
for real, pred, g_, b_, cats, folder, lv, feats in worst:
    print(f'  {folder[:35]:35s} [{lv}] 真实={real:5.2f} 预测={pred:5.2f} d={pred-real:+.2f} '
          f'nps={feats.get("real_core_notes_per_second",0):.2f} dur={feats.get("duration_sec",0):.0f}s')
