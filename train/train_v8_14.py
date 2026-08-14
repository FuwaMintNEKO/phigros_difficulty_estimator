"""v8.14: LightGBM 对比 XGBoost 和 GB"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os, pickle, numpy as np, math, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

print("=" * 60)
print("  v8.14 — LightGBM vs XGBoost vs GB")
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
print(f'特征数: {len(FNo)}')

# 分层分割
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

# GB
gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, min_samples_leaf=3, subsample=0.8, random_state=42)
gb.fit(X_train, train_targets)
gb_preds = gb.predict(X_test)
gb_mae = mean_absolute_error(test_targets, gb_preds)

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1)
xgb_model.fit(X_train, train_targets)
xgb_preds = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(test_targets, xgb_preds)

# LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1, verbose=-1)
lgb_model.fit(X_train, train_targets)
lgb_preds = lgb_model.predict(X_test)
lgb_mae = mean_absolute_error(test_targets, lgb_preds)

print(f'\nGB   MAE: {gb_mae:.4f}')
print(f'XGB  MAE: {xgb_mae:.4f}')
print(f'LGB  MAE: {lgb_mae:.4f}')

# LightGBM 网格搜索
print('\n===== LightGBM 网格搜索 =====')
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.03, 0.05],
    'num_leaves': [15, 31],
    'subsample': [0.8, 1.0],
}
lgb_grid = GridSearchCV(
    lgb.LGBMRegressor(colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1, verbose=-1),
    param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=1, verbose=1
)
lgb_grid.fit(X_train, train_targets)
print(f'最优参数: {lgb_grid.best_params_}')
print(f'CV MAE: {-lgb_grid.best_score_:.4f}')

best_lgb = lgb_grid.best_estimator_
lgb_best_preds = best_lgb.predict(X_test)
lgb_best_mae = mean_absolute_error(test_targets, lgb_best_preds)
print(f'测试集 MAE: {lgb_best_mae:.4f}')

# 按区间对比
print('\n===== 区间对比 =====')
print(f'{"区间":<12s} {"n":>3s} {"GB":>8s} {"XGB":>8s} {"LGB":>8s}')
for lo, hi in [(13,14),(14,15),(15,16),(16,17),(17,20)]:
    mask = (test_targets >= lo) & (test_targets < hi)
    if mask.sum() == 0: continue
    gbm = mean_absolute_error(test_targets[mask], gb_preds[mask])
    xbm = mean_absolute_error(test_targets[mask], xgb_preds[mask])
    lbm = mean_absolute_error(test_targets[mask], lgb_best_preds[mask])
    print(f'[{lo},{hi})      {mask.sum():>3d} {gbm:>8.4f} {xbm:>8.4f} {lbm:>8.4f}')

# 极端谱面
print('\n===== 极端谱面: GB vs XGB vs LGB =====')
print(f'{"Name":<35s} {"True":>6s} {"GB":>8s} {"XGB":>8s} {"LGB":>8s}')
for i in np.argsort(np.abs(test_targets - gb_preds))[-15:]:
    c = test_charts[i]
    name = c.get('_name', '?')[:35]
    true = test_targets[i]
    print(f'{name:<35s} {true:>6.1f} {gb_preds[i]:>8.2f} {xgb_preds[i]:>8.2f} {lgb_best_preds[i]:>8.2f}')

print('\n===== 完成 =====')