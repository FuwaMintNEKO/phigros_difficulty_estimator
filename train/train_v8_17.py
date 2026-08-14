"""v8.17: Stacking — 残差元模型修正极端谱面误差"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os, pickle, numpy as np, math, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

print("=" * 60)
print("  v8.17 — Stacking: 残差元模型")
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

# Level 1: XGBoost base
xgb_params = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1)
xgb_base = xgb.XGBRegressor(**xgb_params)
xgb_base.fit(X_train, train_targets)
base_preds = xgb_base.predict(X_test)
base_mae = mean_absolute_error(test_targets, base_preds)
print(f'XGB Base MAE: {base_mae:.4f}')

# 用交叉验证获取训练集的out-of-fold预测（避免过拟合）
print('Cross-val training residuals...')
oof_preds = cross_val_predict(xgb_base, X_train, train_targets, cv=5)
train_residuals = train_targets - oof_preds
print(f'OOF MAE: {mean_absolute_error(train_targets, oof_preds):.4f}')

# Level 2: 残差模型
# 只用高难度谱面训练残差模型
high_mask = train_targets >= 14.5
print(f'高难度训练样本: {high_mask.sum()}')

# 使用简单特征训练残差模型
# 特征: base_prediction, difficulty, 和几个关键特征
print('\n===== 残差模型对比 =====')

# 方法1: 直接用base_pred作为特征
# 方法2: base_pred + 原始特征
# 方法3: 只用高难度谱面

# 构建残差模型的输入特征
# 用base_pred和前20个重要特征
importances = xgb_base.feature_importances_
top20_idx = np.argsort(importances)[-20:]
top20_feat = [FNo[i] for i in top20_idx]

# 为残差模型构建特征
# 残差模型特征: 只用base_pred和原始特征，不用target（避免数据泄露）
X_res_train = np.column_stack([
    oof_preds,
    X_train[:, top20_idx],
])
X_res_test = np.column_stack([
    base_preds,
    X_test[:, top20_idx],
])

# 尝试不同残差模型
models = [
    ('RF', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)),
    ('XGB', xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.03, random_state=42, n_jobs=1)),
    ('GB', GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.03, random_state=42)),
]

print(f'{"Model":<10s} {"Base MAE":>10s} {"Corrected MAE":>14s} {"High MAE":>10s}')
for name, res_model in models:
    res_model.fit(X_res_train, train_residuals)
    res_preds = res_model.predict(X_res_test)
    corrected = base_preds + res_preds
    
    corrected_mae = mean_absolute_error(test_targets, corrected)
    high_mae = mean_absolute_error(test_targets[test_targets >= 15], corrected[test_targets >= 15])
    
    print(f'{name:<10s} {base_mae:>10.4f} {corrected_mae:>14.4f} {high_mae:>10.4f}')

# 最佳残差模型详细分析
best_res = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.03, random_state=42)
best_res.fit(X_res_train, train_residuals)
best_res_preds = best_res.predict(X_res_test)
best_corrected = base_preds + best_res_preds
best_corrected_mae = mean_absolute_error(test_targets, best_corrected)

print(f'\n===== 极端谱面: Base vs Corrected =====')
print(f'{"Name":<35s} {"True":>6s} {"Base":>8s} {"Corr":>8s} {"ΔErr":>7s}')
for i in np.argsort(np.abs(test_targets - base_preds))[-15:]:
    c = test_charts[i]
    name = c.get('_name', '?')[:35]
    true = test_targets[i]
    old_err = abs(true - base_preds[i])
    new_err = abs(true - best_corrected[i])
    d = new_err - old_err
    sign = '+' if d > 0 else ''
    print(f'{name:<35s} {true:>6.1f} {base_preds[i]:>8.2f} {best_corrected[i]:>8.2f} {sign}{d:>+6.2f}')

print(f'\n===== 完成 =====')