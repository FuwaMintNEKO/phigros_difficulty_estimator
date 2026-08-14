# -*- coding: utf-8 -*-
"""验证核心假设：v9.0 boost (MANUAL_FLAT) 与 v7 训练用的 FLAT_FEATURES 是否匹配"""
import os, sys, json, pickle
import numpy as np

ROOT = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator'
sys.path.insert(0, ROOT)
from feature_extractor import extract_features
from unified_parser import load_chart_from_bytes
from data_loader import load_difficulty_tsv, find_chart_files

with open(os.path.join(ROOT, 'models', '6dim_model_v7.pkl'), 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']
V7_FLAT = m['FLAT_FEATURES']  # v7 训练用的 boost 特征

import importlib.util
spec = importlib.util.spec_from_file_location('appmod', os.path.join(ROOT, 'app.py'))
appmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(appmod)

# v7 训练时的 boost 计算（复刻 train_6dim_v7.py 的 _compute_dim_boost + _dynamic_cap）
DC = m.get('dynamic_cap', {'knee': 1.0, 'power': 0.90})

def v7_boost(feats):
    raw = 0.0
    for fname, baseline, coeff in V7_FLAT:
        val = feats.get(fname, 0)
        pv = P95.get(fname, 0)
        thresh = max(pv * 0.55, baseline * 0.5)
        if val <= thresh:
            continue
        excess = val / thresh - 1.0
        contrib = coeff * (excess ** 0.70)
        if val > max(P99.get(fname, 0), baseline * 0.5):
            p99_excess = val / max(P99.get(fname, 0), baseline * 0.5) - 1.0
            contrib += coeff * max(0, p99_excess) ** 0.70 * 0.5
        raw += contrib
    knee = DC['knee']; power = DC['power']
    if raw <= knee:
        return raw
    return knee + (raw - knee) ** power

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
            rows.append((diffs[sid][lv], lv, feats))
        except Exception:
            pass

print(f'官谱 {len(rows)} 个')
# 分难度统计 v7_boost vs v9_boost
import collections
by_level = collections.defaultdict(list)
for real, lv, feats in rows:
    b7 = v7_boost(feats)
    b9, _, _ = appmod.compute_boost(feats)
    by_level[lv].append((real, b7, b9, feats))

for lv in ['EZ', 'HD', 'IN', 'AT']:
    items = by_level.get(lv, [])
    if not items:
        continue
    reals = [r for r, _, _, _ in items]
    b7s = np.array([b for _, b, _, _ in items])
    b9s = np.array([b for _, _, b, _ in items])
    print(f'  [{lv}] n={len(items):3d}  真实定数avg={np.mean(reals):6.2f}  '
          f'v7_boost avg={np.mean(b7s):5.2f}  v9_boost avg={np.mean(b9s):5.2f}  差值(v9-v7)={np.mean(b9s-b7s):+.3f}')

# 预测 vs 真实 分难度
print()
print('分难度预测对比 (v9 推理逻辑: gb残差 + v9_boost):')
for lv in ['EZ', 'HD', 'IN', 'AT']:
    items = by_level.get(lv, [])
    if not items:
        continue
    errs_v9, errs_v7 = [], []
    for real, b7, b9, feats in items:
        x = np.array([[feats.get(n, 0) for n in FN]])
        xs = scaler.transform(x)
        p_gb = float(gb.predict(xs)[0])
        errs_v9.append(p_gb + b9 - real)
        errs_v7.append(p_gb + b7 - real)
    errs_v9 = np.array(errs_v9); errs_v7 = np.array(errs_v7)
    print(f'  [{lv}] n={len(errs_v9):3d}  v9: Bias={np.mean(errs_v9):+.3f} MAE={np.mean(np.abs(errs_v9)):.3f}  |  v7: Bias={np.mean(errs_v7):+.3f} MAE={np.mean(np.abs(errs_v7)):.3f}')
