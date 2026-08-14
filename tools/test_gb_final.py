import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

base_chart = os.path.join(_ROOT, 'data', 'chart')
tsv = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

diffs = load_difficulty_tsv(tsv)
charts = find_chart_files(base_chart)

all_feats, all_labels, all_levels = [], [], []
for folder, info in charts.items():
    sid = info['song_id']
    if sid not in diffs: continue
    for level in ['EZ', 'HD', 'IN', 'AT']:
        if level not in info['levels'] or level not in diffs[sid]: continue
        try:
            data = load_chart_json(info['levels'][level])
            feats = extract_features(data)
            if feats is not None:
                all_feats.append(feats)
                all_labels.append(diffs[sid][level])
                all_levels.append(level)
        except: pass

feature_names = list(all_feats[0].keys())
X = np.array([[f.get(n, 0) for n in feature_names] for f in all_feats])
y = np.array(all_labels)
levels = np.array(all_levels)

print(f'特征数: {len(feature_names)}')
print(f'样本: {len(y)} (AT={sum(levels=="AT")}, IN={sum(levels=="IN")}, HD={sum(levels=="HD")}, EZ={sum(levels=="EZ")})')

# Train on all data with weights
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

weights = np.ones(len(y))
for i, lv in enumerate(levels):
    if lv == 'AT': weights[i] = 30.0
    elif lv == 'IN': weights[i] = 2.0

gb = GradientBoostingRegressor(n_estimators=700, learning_rate=0.06, max_depth=6, min_samples_leaf=3, subsample=0.85, random_state=42)
gb.fit(X_s, y, sample_weight=weights)

pred = gb.predict(X_s)

at_mask = levels == 'AT'
ezhd_mask = levels != 'AT'

print(f'\n=== GB单模型(全量训练) ===')
print(f'AT R² = {r2_score(y[at_mask], pred[at_mask]):.4f}')
print(f'AT MAE = {mean_absolute_error(y[at_mask], pred[at_mask]):.4f}')
print(f'EZ+HD+IN R² = {r2_score(y[ezhd_mask], pred[ezhd_mask]):.4f}')
print(f'Overall R² = {r2_score(y, pred):.4f}')

# Cross-validation for fair evaluation (stratified by level)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

at_r2_scores = []
all_r2_scores = []
for train_idx, test_idx in skf.split(X, levels):
    X_t, X_te = X[train_idx], X[test_idx]
    y_t, y_te = y[train_idx], y[test_idx]
    l_te = levels[test_idx]
    
    scl = StandardScaler()
    X_t_s = scl.fit_transform(X_t)
    X_te_s = scl.transform(X_te)
    
    w = np.ones(len(y_t))
    for j, lv in enumerate(levels[train_idx]):
        if lv == 'AT': w[j] = 30.0
        elif lv == 'IN': w[j] = 2.0
    
    gb_cv = GradientBoostingRegressor(n_estimators=700, learning_rate=0.06, max_depth=6, min_samples_leaf=3, subsample=0.85, random_state=42)
    gb_cv.fit(X_t_s, y_t, sample_weight=w)
    p = gb_cv.predict(X_te_s)
    
    at_r2_scores.append(r2_score(y_te[l_te=='AT'], p[l_te=='AT']))
    all_r2_scores.append(r2_score(y_te, p))

print(f'\n=== 5折交叉验证(分层) ===')
print(f'AT R² = {np.mean(at_r2_scores):.4f} (±{np.std(at_r2_scores):.4f})')
print(f'Overall R² = {np.mean(all_r2_scores):.4f} (±{np.std(all_r2_scores):.4f})')

# AT details
print(f'\nAT 谱面详情:')
at_indices = np.where(at_mask)[0]
at_info = []
for folder, info in charts.items():
    sid = info['song_id']
    if sid not in diffs: continue
    if 'AT' in info['levels'] and 'AT' in diffs[sid]:
        for idx in at_indices:
            idx_chart = list(charts.keys())[list(charts.values()).index(info)]
            at_name = folder.replace('.0', '')
            at_info.append((at_name, diffs[sid]['AT'], pred[idx]))
            break

# Rebuild properly
at_info2 = []
folders_list = list(charts.keys())
idx = 0
for i, folder in enumerate(folders_list):
    info = charts[folder]
    sid = info['song_id']
    if sid in diffs and 'AT' in info['levels'] and 'AT' in diffs[sid]:
        at_info2.append((folder.replace('.0', ''), diffs[sid]['AT'], pred[at_indices[idx]]))
        idx += 1

at_info2.sort(key=lambda x: x[0].lower())
print(f"{'序':>3s}  {'谱面名称':<55s}  {'预测':>8s}  {'真实':>6s}  {'偏差':>7s}")
print('-' * 85)
for rank, (name, actual, p) in enumerate(at_info2, 1):
    err = p - actual
    print(f"{rank:>3d}  {name[:55]:<55s}  {p:>8.2f}  {actual:>6.2f}  {err:>+7.2f}")
