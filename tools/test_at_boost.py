import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np

base_chart = os.path.join(_ROOT, 'data', 'chart')
tsv = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

diffs = load_difficulty_tsv(tsv)
charts = find_chart_files(base_chart)

all_feats, all_labels, all_levels, all_names = [], [], [], []
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
                all_names.append(f'{folder.replace(".0","")}_{level}')
        except: pass

feature_names = list(all_feats[0].keys())
X = np.array([[f.get(n, 0) for n in feature_names] for f in all_feats])
y = np.array(all_labels)
levels = np.array(all_levels)

# 1) Split test set FIRST (no contamination)
train_idx, test_idx = train_test_split(
    range(len(y)), test_size=0.15, random_state=42,
    stratify=levels
)

X_train = X[train_idx]
y_train = y[train_idx]
l_train = levels[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]
l_test = levels[test_idx]

print(f'Train: {len(X_train)}, Test: {len(X_test)}')
print(f'AT in train: {sum(l_train=="AT")}, AT in test: {sum(l_test=="AT")}')

# 2) Oversample AT in training set only
at_train_idx = np.where(l_train == 'AT')[0]
oversample_factor = 12
X_train_aug = list(X_train)
y_train_aug = list(y_train)
l_train_aug = list(l_train)
for _ in range(oversample_factor):
    for i in at_train_idx:
        noise = np.random.normal(0, 0.001, X_train[i].shape)
        X_train_aug.append(X_train[i] + noise)
        y_train_aug.append(y_train[i])
        l_train_aug.append('AT')

X_train_aug = np.array(X_train_aug)
y_train_aug = np.array(y_train_aug)
l_train_aug = np.array(l_train_aug)
print(f'Augmented train: {len(X_train_aug)} (AT x{oversample_factor})')

# 3) Train with weights
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_aug)
X_test_s = scaler.transform(X_test)

weights = np.ones(len(y_train_aug))
for i, lv in enumerate(l_train_aug):
    if lv == 'AT': weights[i] = 8.0
    elif lv == 'IN': weights[i] = 1.8

gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.08, max_depth=6, min_samples_leaf=3, random_state=42)
gb.fit(X_train_s, y_train_aug, sample_weight=weights)

rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train_aug, sample_weight=weights)

# 4) Evaluate on entire dataset
X_all = scaler.transform(X)
pred_gb = gb.predict(X_all)
pred_rf = rf.predict(X_all)
pred_ens = (pred_gb + pred_rf) / 2

at_mask = levels == 'AT'
ezhd_mask = levels != 'AT'

print('\n' + '=' * 70)
print('  全统一模型 (AT过采样12x + 加权)')
print('=' * 70)
print(f'\nAT R² GB:  {r2_score(y[at_mask], pred_gb[at_mask]):.4f}')
print(f'AT R² RF:  {r2_score(y[at_mask], pred_rf[at_mask]):.4f}')
print(f'AT R² Ens: {r2_score(y[at_mask], pred_ens[at_mask]):.4f}')
print(f'EZ+HD+IN R² GB:  {r2_score(y[ezhd_mask], pred_gb[ezhd_mask]):.4f}')
print(f'EZ+HD+IN R² RF:  {r2_score(y[ezhd_mask], pred_rf[ezhd_mask]):.4f}')
print(f'EZ+HD+IN R² Ens: {r2_score(y[ezhd_mask], pred_ens[ezhd_mask]):.4f}')
print(f'\nOverall R² GB:  {r2_score(y, pred_gb):.4f}')
print(f'Overall R² RF:  {r2_score(y, pred_rf):.4f}')
print(f'Overall R² Ens: {r2_score(y, pred_ens):.4f}')

print(f'\nTest set only R²:')
test_at = l_test == 'AT'
test_ezhd = l_test != 'AT'
pred_test_gb = gb.predict(X_test_s)
print(f'  AT(GB): {r2_score(y_test[test_at], pred_test_gb[test_at]):.4f} [{sum(test_at)} samples]')
print(f'  EZ+HD+IN(GB): {r2_score(y_test[test_ezhd], pred_test_gb[test_ezhd]):.4f}')

# 5) Print AT details sorted by name
print(f'\nAT 谱面预测详情 (GB单模型):')
at_indices = np.where(at_mask)[0]
at_info = []
for i in at_indices:
    name = all_names[i].rsplit('_', 1)[0]
    at_info.append((name, y[i], pred_gb[i], pred_rf[i], pred_ens[i]))

at_info.sort(key=lambda x: x[0].lower())

print(f"{'序':>3s}  {'谱面名称':<55s}  {'预测(GB)':>8s}  {'真实':>6s}  {'偏差':>7s}  {'Ens':>6s}")
print('-' * 95)
for rank, (name, actual, pred_g, pred_r, pred_e) in enumerate(at_info, 1):
    err = pred_g - actual
    n = name[:55]
    print(f"{rank:>3d}  {n:<55s}  {pred_g:>8.2f}  {actual:>6.2f}  {err:>+7.2f}  {pred_e:>6.2f}")

print(f'\nAT GB R² = {r2_score(y[at_mask], pred_gb[at_mask]):.4f}')
