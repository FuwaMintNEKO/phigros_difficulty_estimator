"""v8.16: 特征选择 + 加权训练 — 改善极端谱面预测"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os, pickle, numpy as np, math, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

print("=" * 60)
print("  v8.16 — 特征选择 + 加权训练")
print("=" * 60)

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)

all_charts = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    for lv in ['IN', 'AT']:
        if lv not in info.get('levels', {}): continue
        if lv not in song_difficulties[sid]: continue
        try:
            cd = load_chart_json(info['levels'][lv])
            feats = extract_features(cd)
            if feats:
                feats['_difficulty'] = song_difficulties[sid][lv]
                feats['_name'] = fn[:30]
                all_charts.append(feats)
        except Exception as e: pass

exclude_patterns = ['snowmelt', 'snowdance', 'snow dance']
all_charts = [f for f in all_charts if not any(p.lower() in f['_name'].lower() for p in exclude_patterns)]
print(f'总谱面数: {len(all_charts)}')

FNo = sorted({k for f in all_charts for k in f.keys() if not k.startswith('_')})

diffs = np.array([f['_difficulty'] for f in all_charts])
bins = np.digitize(diffs, bins=[13, 14, 15, 16, 17])
train_mask = np.zeros(len(all_charts), dtype=bool)
test_mask = np.zeros(len(all_charts), dtype=bool)
np.random.seed(42)
for b in range(1, 6):
    idx = np.where(bins == b)[0]
    if len(idx) == 0: continue
    tr_idx, te_idx = train_test_split(idx, test_size=0.25, random_state=42)
    train_mask[tr_idx] = True
    test_mask[te_idx] = True

train_charts = [all_charts[i] for i in range(len(all_charts)) if train_mask[i]]
test_charts = [all_charts[i] for i in range(len(all_charts)) if test_mask[i]]
print(f'训练集: {len(train_charts)} 谱面, 测试集: {len(test_charts)} 谱面')

train_targets = np.array([f['_difficulty'] for f in train_charts])
test_targets = np.array([f['_difficulty'] for f in test_charts])
X_train = np.array([[f.get(n, 0) for n in FNo] for f in train_charts])
X_test = np.array([[f.get(n, 0) for n in FNo] for f in test_charts])

# Baseline
xgb_base = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1)
xgb_base.fit(X_train, train_targets)
base_preds = xgb_base.predict(X_test)
base_mae = mean_absolute_error(test_targets, base_preds)
print(f'Baseline MAE: {base_mae:.4f}')

# 获取特征重要性，选择Top N
importances = xgb_base.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
cumsum = np.cumsum(importances[sorted_idx])

# 找到累积重要性达到95%的特征数
n95 = np.searchsorted(cumsum, 0.95) + 1
print(f'累积重要性95%: {n95} 个特征 (共{len(FNo)})')

# 测试不同特征数
print('\n===== 特征选择 =====')
print(f'{"N":>5s} {"Cum%":>7s} {"MAE":>8s} {"Best10":>8s}')
best_feat_mae = base_mae
best_feat_n = len(FNo)

for n in [50, 75, 100, 125, 150, 200, n95, 250]:
    selected = [FNo[i] for i in sorted_idx[:n]]
    X_tr = np.array([[f.get(k, 0) for k in selected] for f in train_charts])
    X_te = np.array([[f.get(k, 0) for k in selected] for f in test_charts])
    
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1)
    m.fit(X_tr, train_targets)
    preds = m.predict(X_te)
    mae = mean_absolute_error(test_targets, preds)
    
    worst_idx = np.argsort(np.abs(test_targets - base_preds))[-10:]
    b10 = mean_absolute_error(test_targets[worst_idx], preds[worst_idx])
    
    cum = cumsum[n-1] if n <= len(cumsum) else 1.0
    marker = ' <-- BEST' if mae < best_feat_mae else ''
    print(f'{n:>5d} {cum*100:>6.1f}% {mae:>8.4f} {b10:>8.4f}{marker}')
    if mae < best_feat_mae:
        best_feat_mae = mae
        best_feat_n = n

# 加权训练：高难度谱面更高权重
print('\n===== 加权训练 =====')
weights = np.ones_like(train_targets)
# 对高难度谱面加权
weights[train_targets >= 15] = 2.0
weights[train_targets >= 16] = 3.0

xgb_weighted = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1)
xgb_weighted.fit(X_train, train_targets, sample_weight=weights)
w_preds = xgb_weighted.predict(X_test)
w_mae = mean_absolute_error(test_targets, w_preds)
print(f'加权 MAE: {w_mae:.4f} (vs baseline {base_mae:.4f})')

# 按区间看加权效果
print(f'\n{"区间":<12s} {"n":>3s} {"Baseline":>8s} {"Weighted":>8s} {"Δ":>7s}')
for lo, hi in [(13,14),(14,15),(15,16),(16,17),(17,20)]:
    mask = (test_targets >= lo) & (test_targets < hi)
    if mask.sum() == 0: continue
    bm = mean_absolute_error(test_targets[mask], base_preds[mask])
    wm = mean_absolute_error(test_targets[mask], w_preds[mask])
    d = wm - bm
    print(f'[{lo},{hi})      {mask.sum():>3d} {bm:>8.4f} {wm:>8.4f} {d:>+7.4f}')

# 高难度谱面对比
print(f'\n===== 高难度谱面 (≥15.5): Baseline vs Weighted =====')
print(f'{"Name":<35s} {"True":>6s} {"Base":>8s} {"W":>8s}')
for i in np.argsort(test_targets)[-15:]:
    c = test_charts[i]
    name = c.get('_name', '?')[:35]
    true = test_targets[i]
    print(f'{name:<35s} {true:>6.1f} {base_preds[i]:>8.2f} {w_preds[i]:>8.2f}')

print('\n===== 完成 =====')