"""v8.20: 分段模型 — 按难度区间分别训练"""
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
print("  v8.20 — 分段模型 (By Difficulty Range)")
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

# 单模型基线
xgb_base = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1)
xgb_base.fit(X_train, train_targets)
base_preds = xgb_base.predict(X_test)
base_mae = mean_absolute_error(test_targets, base_preds)
print(f'单一XGB MAE: {base_mae:.4f}')

# 分段模型：按难度分别训练
ranges = [(13, 14.5), (14.5, 16), (16, 20)]
print(f'\n===== 分段模型 =====')
models = {}
all_preds = np.zeros(len(test_charts))

for lo, hi in ranges:
    tr_mask = (train_targets >= lo) & (train_targets < hi)
    te_mask = (test_targets >= lo) & (test_targets < hi)
    
    if tr_mask.sum() < 10:
        print(f'[{lo},{hi}) 训练样本不足 ({tr_mask.sum()}), 跳过')
        continue
    
    m = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1)
    m.fit(X_train[tr_mask], train_targets[tr_mask])
    
    if te_mask.sum() > 0:
        preds = m.predict(X_test[te_mask])
        all_preds[te_mask] = preds
        mae = mean_absolute_error(test_targets[te_mask], preds)
        single_mae = mean_absolute_error(test_targets[te_mask], base_preds[te_mask])
        print(f'[{lo},{hi}) 训练{tr_mask.sum():>3d} 测试{te_mask.sum():>2d} | 单一: {single_mae:.4f} | 分段: {mae:.4f}')
    
    models[(lo, hi)] = m

# 处理未覆盖的测试样本
missing = all_preds == 0
if missing.any():
    all_preds[missing] = base_preds[missing]

segmented_mae = mean_absolute_error(test_targets, all_preds)
print(f'\n分段模型总MAE: {segmented_mae:.4f} (vs 单一: {base_mae:.4f})')

# 极端谱面
print(f'\n===== 极端谱面: 单一 vs 分段 =====')
print(f'{"Name":<35s} {"True":>6s} {"单一":>8s} {"分段":>8s}')
for i in np.argsort(np.abs(test_targets - base_preds))[-15:]:
    c = test_charts[i]
    name = c.get('_name', '?')[:35]
    true = test_targets[i]
    print(f'{name:<35s} {true:>6.1f} {base_preds[i]:>8.2f} {all_preds[i]:>8.2f}')

print('\n===== 完成 =====')