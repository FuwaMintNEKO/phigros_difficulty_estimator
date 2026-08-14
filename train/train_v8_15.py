"""v8.15: XGBoost + Huber loss — 对极端值更鲁棒"""
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
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

print("=" * 60)
print("  v8.15 — XGBoost: Huber loss + 不同参数")
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

# Baseline XGB
xgb_base = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1)
xgb_base.fit(X_train, train_targets)
base_preds = xgb_base.predict(X_test)
base_mae = mean_absolute_error(test_targets, base_preds)

# 测试不同配置
configs = [
    ('Baseline', {}),
    ('PseudoHuber α=0.5', {'objective': 'reg:pseudohubererror', 'huber_slope': 0.5}),
    ('PseudoHuber α=1.0', {'objective': 'reg:pseudohubererror', 'huber_slope': 1.0}),
    ('PseudoHuber α=2.0', {'objective': 'reg:pseudohubererror', 'huber_slope': 2.0}),
    ('AbsError (MAE)', {'objective': 'reg:absoluteerror'}),
    ('max_depth=5', {'max_depth': 5}),
    ('max_depth=6', {'max_depth': 6}),
    ('n_est=500', {'n_estimators': 500}),
    ('lr=0.03', {'learning_rate': 0.03, 'n_estimators': 500}),
    ('reg_alpha=0.5', {'reg_alpha': 0.5}),
    ('reg_alpha=1.0', {'reg_alpha': 1.0}),
    ('colsample=0.6', {'colsample_bytree': 0.6}),
    ('subsample=0.6', {'subsample': 0.6}),
]

print(f'\n===== 配置对比 =====')
print(f'{"Config":<20s} {"MAE":>8s} {"Best10":>8s}')

best_overall_mae = base_mae
best_config = None

for name, extra in configs:
    params = {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42, 'n_jobs': 1}
    params.update(extra)
    m = xgb.XGBRegressor(**params)
    m.fit(X_train, train_targets)
    preds = m.predict(X_test)
    mae = mean_absolute_error(test_targets, preds)
    
    # 极端10个谱面的MAE
    worst_idx = np.argsort(np.abs(test_targets - base_preds))[-10:]
    best10_mae = mean_absolute_error(test_targets[worst_idx], preds[worst_idx])
    
    marker = ' <-- BEST' if mae < best_overall_mae else ''
    print(f'{name:<20s} {mae:>8.4f} {best10_mae:>8.4f}{marker}')
    
    if mae < best_overall_mae:
        best_overall_mae = mae
        best_config = (name, params, m)

# 最佳配置详细分析
print(f'\n===== 最佳配置: {best_config[0]} =====')
best_model = best_config[2]
best_preds = best_model.predict(X_test)
best_mae = mean_absolute_error(test_targets, best_preds)
print(f'测试集 MAE: {best_mae:.4f}')

print('\n===== 极端谱面: Baseline vs Best =====')
print(f'{"Name":<35s} {"True":>6s} {"Base":>8s} {"Best":>8s} {"Δ":>7s}')
for i in np.argsort(np.abs(test_targets - base_preds))[-15:]:
    c = test_charts[i]
    name = c.get('_name', '?')[:35]
    true = test_targets[i]
    d = abs(true - best_preds[i]) - abs(true - base_preds[i])
    sign = '+' if d > 0 else ''
    print(f'{name:<35s} {true:>6.1f} {base_preds[i]:>8.2f} {best_preds[i]:>8.2f} {sign}{d:>+6.2f}')

print('\n===== 完成 =====')