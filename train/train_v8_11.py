"""v8.11: XGBoost 对比 GB — 看能否改善极端谱面误差"""
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
import xgboost as xgb

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

print("=" * 60)
print("  v8.11 — XGBoost 对比 GB")
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

# GB baseline (v8.9最优参数)
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, min_samples_leaf=3, subsample=0.8, random_state=42)
gb.fit(X_train, train_targets)
gb_preds = gb.predict(X_test)
gb_mae = mean_absolute_error(test_targets, gb_preds)
print(f'\nGB MAE: {gb_mae:.4f}')

# XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1)
xgb_model.fit(X_train, train_targets)
xgb_preds = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(test_targets, xgb_preds)
print(f'XGB MAE: {xgb_mae:.4f}')

# 网格搜索 XGBoost
print('\n===== XGBoost 网格搜索 =====')
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.03, 0.05],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}
xgb_grid = GridSearchCV(
    xgb.XGBRegressor(reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1),
    param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=1, verbose=1
)
xgb_grid.fit(X_train, train_targets)
print(f'最优参数: {xgb_grid.best_params_}')
print(f'CV MAE: {-xgb_grid.best_score_:.4f}')

best_xgb = xgb_grid.best_estimator_
xgb_best_preds = best_xgb.predict(X_test)
xgb_best_mae = mean_absolute_error(test_targets, xgb_best_preds)
print(f'测试集 MAE: {xgb_best_mae:.4f}')

# 按区间对比
print('\n===== 区间对比 =====')
print(f'{"区间":<12s} {"n":>3s} {"GB":>8s} {"XGB":>8s} {"Δ":>8s}')
for lo, hi in [(13,14),(14,15),(15,16),(16,17),(17,20)]:
    mask = (test_targets >= lo) & (test_targets < hi)
    if mask.sum() == 0: continue
    gbm = mean_absolute_error(test_targets[mask], gb_preds[mask])
    xbm = mean_absolute_error(test_targets[mask], xgb_best_preds[mask])
    d = xbm - gbm
    sign = '+' if d > 0 else ''
    print(f'[{lo},{hi})      {mask.sum():>3d} {gbm:>8.4f} {xbm:>8.4f} {sign}{d:>7.4f}')

# 极端谱面对比
print('\n===== 极端谱面: GB vs XGB =====')
print(f'{"Name":<35s} {"True":>6s} {"GB":>8s} {"XGB":>8s} {"Δ":>7s}')
for i in np.argsort(np.abs(test_targets - gb_preds))[-15:]:
    c = test_charts[i]
    name = c.get('_name', '?')[:35]
    true = test_targets[i]
    gb_p = gb_preds[i]
    xgb_p = xgb_best_preds[i]
    d = abs(true - xgb_p) - abs(true - gb_p)
    sign = '+' if d > 0 else ''
    print(f'{name:<35s} {true:>6.1f} {gb_p:>8.2f} {xgb_p:>8.2f} {sign}{d:>+6.2f}')

# 全量训练
print('\n===== 全量训练 =====')
X_all = np.array([[f.get(n, 0) for n in FNo] for f in all_charts])
y_all = np.array([f['_difficulty'] for f in all_charts])
best_xgb.fit(X_all, y_all)
y_pred = best_xgb.predict(X_all)
full_mae = mean_absolute_error(y_all, y_pred)
print(f'全量 MAE: {full_mae:.4f}')

# 保存
with open('models/6dim_model_v8_9.pkl', 'rb') as f:
    m89 = pickle.load(f)

model = {
    'gb': best_xgb, 'scaler': StandardScaler().fit(X_all),  # XGB不需要scaler，但保持接口一致
    'feature_names': FNo,
    'p95_vals': m89['p95_vals'], 'p99_vals': m89['p99_vals'],
    'FLAT_FEATURES': m89['FLAT_FEATURES'],
    'dynamic_cap': m89['dynamic_cap'],
    'version': '8.11',
    'model_type': 'xgboost',
    'scaler': StandardScaler().fit(X_all),  # 占位，XGB不需要标准化
}
with open('models/6dim_model_v8_11.pkl', 'wb') as f:
    pickle.dump(model, f)
print('Saved: models/6dim_model_v8_11.pkl')
print('\n===== 完成 =====')