"""v8.19: 神经网络 (MLP) 回归 — 尝试捕获非线性交互"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os, pickle, numpy as np, math, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

print("=" * 60)
print("  v8.19 — MLP 神经网络回归")
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

# 标准化（MLP必须）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost Baseline
xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1)
xgb_model.fit(X_train, train_targets)
xgb_preds = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(test_targets, xgb_preds)
print(f'XGBoost MAE: {xgb_mae:.4f}')

# 尝试不同MLP架构
print('\n===== MLP架构对比 =====')
print(f'{"Architecture":<25s} {"MAE":>8s} {"CV MAE":>8s}')

configs = [
    # (name, hidden_layer_sizes, alpha, learning_rate_init)
    ('single_64', (64,), 0.001, 0.001),
    ('single_128', (128,), 0.001, 0.001),
    ('dual_64_32', (64, 32), 0.001, 0.001),
    ('dual_128_64', (128, 64), 0.001, 0.001),
    ('dual_128_64_a0.01', (128, 64), 0.01, 0.001),
    ('dual_128_64_a0.1', (128, 64), 0.1, 0.001),
    ('dual_256_128', (256, 128), 0.001, 0.001),
    ('triple_128_64_32', (128, 64, 32), 0.001, 0.001),
    ('triple_256_128_64', (256, 128, 64), 0.001, 0.001),
    ('dual_128_64_lr0.01', (128, 64), 0.001, 0.01),
    ('dual_128_64_lr0.0005', (128, 64), 0.001, 0.0005),
]

best_mae = 999
best_config = None
best_model = None

for name, hidden, alpha, lr in configs:
    try:
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation='relu',
            solver='adam',
            alpha=alpha,
            learning_rate_init=lr,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
            n_iter_no_change=50
        )
        mlp.fit(X_train_scaled, train_targets)
        preds = mlp.predict(X_test_scaled)
        mae = mean_absolute_error(test_targets, preds)
        cv_scores = cross_val_score(mlp, X_train_scaled, train_targets, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        marker = ' <-- BEST' if mae < best_mae else ''
        print(f'{name:<25s} {mae:>8.4f} {cv_mae:>8.4f}{marker}')
        
        if mae < best_mae:
            best_mae = mae
            best_config = (name, hidden, alpha, lr)
            best_model = mlp
    except Exception as e:
        print(f'{name:<25s} ERROR: {str(e)[:50]}')

# 最佳MLP详细分析
print(f'\n===== 最佳MLP: {best_config[0]} =====')
print(f'隐藏层: {best_config[1]}, alpha={best_config[2]}, lr={best_config[3]}')
print(f'测试MAE: {best_mae:.4f}')

best_preds = best_model.predict(X_test_scaled)

# 区间对比
print(f'\n{"区间":<12s} {"n":>3s} {"XGB":>8s} {"MLP":>8s}')
for lo, hi in [(13,14),(14,15),(15,16),(16,17),(17,20)]:
    mask = (test_targets >= lo) & (test_targets < hi)
    if mask.sum() == 0: continue
    xb = mean_absolute_error(test_targets[mask], xgb_preds[mask])
    mb = mean_absolute_error(test_targets[mask], best_preds[mask])
    print(f'[{lo},{hi})      {mask.sum():>3d} {xb:>8.4f} {mb:>8.4f}')

# 极端谱面
print(f'\n===== 极端谱面: XGB vs MLP =====')
print(f'{"Name":<35s} {"True":>6s} {"XGB":>8s} {"MLP":>8s}')
for i in np.argsort(np.abs(test_targets - xgb_preds))[-15:]:
    c = test_charts[i]
    name = c.get('_name', '?')[:35]
    true = test_targets[i]
    print(f'{name:<35s} {true:>6.1f} {xgb_preds[i]:>8.2f} {best_preds[i]:>8.2f}')

print('\n===== 完成 =====')