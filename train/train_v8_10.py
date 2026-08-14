"""v8.10: 纯GB模型 (无boost特征)，用v8.9网格搜索的最优参数"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os, pickle, numpy as np, math, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

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
print(f"总谱面数: {len(all_charts)}")

# 特征提取
FNo = sorted({k for f in all_charts for k in f.keys() if not k.startswith('_')})
print(f"特征数: {len(FNo)}")

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

# 构建数据
train_targets = np.array([f['_difficulty'] for f in train_charts])
test_targets = np.array([f['_difficulty'] for f in test_charts])

X_train = np.array([[f.get(n, 0) for n in FNo] for f in train_charts])
X_test = np.array([[f.get(n, 0) for n in FNo] for f in test_charts])

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 最优参数 (来自 v8.9 网格搜索)
gb = GradientBoostingRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    min_samples_leaf=3, subsample=0.8, random_state=42
)
gb.fit(X_train_scaled, train_targets)

test_preds = gb.predict(X_test_scaled)
test_mae = mean_absolute_error(test_targets, test_preds)
print(f"测试集 MAE: {test_mae:.4f}")

# 按区间
print("\n===== 测试集按区间 MAE =====")
for lo, hi in [(13,14),(14,15),(15,16),(16,17),(17,20)]:
    mask = (test_targets >= lo) & (test_targets < hi)
    if mask.sum() == 0: continue
    m = mean_absolute_error(test_targets[mask], test_preds[mask])
    print(f"  [{lo},{hi}) n={mask.sum():2d}  MAE={m:.4f}")

# 全量训练
X_all = np.array([[f.get(n, 0) for n in FNo] for f in all_charts])
y_all = np.array([f['_difficulty'] for f in all_charts])
scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X_all)
gb_all = GradientBoostingRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    min_samples_leaf=3, subsample=0.8, random_state=42
)
gb_all.fit(X_all_scaled, y_all)

y_pred = gb_all.predict(X_all_scaled)
full_mae = mean_absolute_error(y_all, y_pred)
print(f"\n全量 MAE: {full_mae:.4f}")

# 加载 P95/P99/FLAT/DC 从 v8.9
with open('models/6dim_model_v8_9.pkl', 'rb') as f:
    m89 = pickle.load(f)

# 保存
model = {
    'gb': gb_all, 'scaler': scaler_all,
    'feature_names': FNo,  # 纯特征，不包含 boost
    'p95_vals': m89['p95_vals'], 'p99_vals': m89['p99_vals'],
    'FLAT_FEATURES': m89['FLAT_FEATURES'],
    'dynamic_cap': m89['dynamic_cap'],
    'version': '8.10',
}
with open('models/6dim_model_v8_10.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved: models/6dim_model_v8_10.pkl")

# 极端谱面诊断
print("\n===== 极端谱面诊断 (测试集) =====")
print(f"{'Name':<35s} {'True':>6s} {'Pred':>8s} {'Err':>7s}")
for i in np.argsort(np.abs(test_targets - test_preds))[-20:]:
    c = test_charts[i]
    name = c.get('_name', '?')[:35]
    true = test_targets[i]
    pred = test_preds[i]
    print(f"{name:<35s} {true:>6.1f} {pred:>8.2f} {pred-true:>+7.2f}")

print("\n===== 完成 =====")