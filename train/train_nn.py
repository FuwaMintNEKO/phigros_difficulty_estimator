import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os
import sys
import json
import pickle
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

print('=' * 60)
print('  神经网络 + KernelRidge 训练')
print('=' * 60)

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)

all_items = []
for folder_name, info in chart_files.items():
    song_id = info['song_id']
    if song_id not in song_difficulties:
        continue
    diffs = song_difficulties[song_id]
    for level in ['EZ', 'HD', 'IN', 'AT']:
        if level in info['levels'] and level in diffs:
            all_items.append({
                'folder': folder_name, 'filepath': info['levels'][level],
                'difficulty': diffs[level], 'level': level,
            })

print(f'\n总样本: {len(all_items)}')

features_list, labels, levels_list, names_list = [], [], [], []
failed = 0
for i, item in enumerate(all_items):
    try:
        chart_data = load_chart_json(item['filepath'])
        feats = extract_features(chart_data)
        if feats is not None:
            features_list.append(feats)
            labels.append(item['difficulty'])
            levels_list.append(item['level'])
            names_list.append(f'{item["folder"]}_{item["level"]}')
        else:
            failed += 1
    except:
        failed += 1
    if (i + 1) % 300 == 0:
        print(f'  已处理 {i+1}/{len(all_items)}...')

print(f'  成功: {len(features_list)}, 失败: {failed}')

feature_names = sorted(features_list[0].keys())
X = np.array([[f.get(n, 0) for n in feature_names] for f in features_list])
y = np.array(labels)
level_arr = np.array(levels_list)
print(f'  特征数: {len(feature_names)}, 难度范围: {y.min():.1f}~{y.max():.1f}')

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分层拆分
difficulty_bins = np.digitize(y, bins=[0, 5, 7, 9, 11, 13, 14, 15, 16, 16.5, 17])
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(sss.split(X, difficulty_bins))
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
level_train, level_test = level_arr[train_idx], level_arr[test_idx]
names_test = [names_list[i] for i in test_idx]
print(f'训练集: {len(X_train)}, 测试集: {len(X_test)}')

# =======================================
# 训练 MLP (神经网络) — 线性输出层可外推
# =======================================
print('\n--- MLPRegressor (256,128,64) ---')
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=3000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=50,
    random_state=42,
    verbose=False,
)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
r2_mlp = r2_score(y_test, y_pred_mlp)
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
print(f'  测试集: R²={r2_mlp:.4f}, MAE={mae_mlp:.4f}')

# =======================================
# 训练 Ridge (线性模型，天然可外推)
# =======================================
print('\n--- Ridge (线性) ---')
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
print(f'  测试集: R²={r2_ridge:.4f}, MAE={mae_ridge:.4f}')

# =======================================
# 集成 MLP + Ridge
# Ridge保证外推能力，MLP保证内推精度
# =======================================
print('\n--- MLP + Ridge 集成 ---')
w_mlp = 0.8
w_ridge = 0.2
w_sum = w_mlp + w_ridge
w_mlp_n, w_ridge_n = w_mlp / w_sum, w_ridge / w_sum
print(f'  集成权重: MLP={w_mlp_n:.3f}, Ridge={w_ridge_n:.3f}')

y_pred_en = y_pred_mlp * w_mlp_n + y_pred_ridge * w_ridge_n
r2_en = r2_score(y_test, y_pred_en)
mae_en = mean_absolute_error(y_test, y_pred_en)
print(f'\n  集成 测试集: R²={r2_en:.4f}, MAE={mae_en:.4f}')

print(f'\n  分难度表现:')
for lv in ['EZ', 'HD', 'IN', 'AT']:
    mask = level_test == lv
    if np.sum(mask) < 3: continue
    y_t = y_test[mask]
    y_p = y_pred_en[mask]
    print(f'    {lv} ({np.sum(mask)}个): R²={r2_score(y_t, y_p):.4f}, MAE={mean_absolute_error(y_t, y_p):.4f}, '
          f'偏差={np.mean(y_p-y_t):+.3f}, ±0.1={np.mean(np.abs(y_p-y_t)<=0.1)*100:.0f}%')

# =======================================
# 预测测试谱面
# =======================================
print('\n' + '=' * 60)
print('  预测测试谱面')
print('=' * 60)

sys.path.insert(0, os.path.dirname(__file__))
from predict_rpe import convert_rpe_to_standard

test_charts = [
    ('Chart_SP', os.path.join(_ROOT, 'data', 'chart', 'Chart_SP.json'), False),
    ('Regrets', os.path.join(_ROOT, 'data', 'chart', 'Sigma (Haocore Mix) ~ Regrets of The Yellow Tuli.json'), False),
    ('105秒伝說', os.path.join(_ROOT, 'data', 'chart', 'Sigma (Haocore Mix) ~ 105秒の伝說 ~.json'), False),
    ('Aether Crest (手速)', os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json'), True),
]

for name, path, is_rpe in test_charts:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    chart_data = convert_rpe_to_standard(raw) if is_rpe else raw

    feats = extract_features(chart_data)
    if feats is None: continue

    x = np.array([[feats.get(n, 0) for n in feature_names]])
    x_s = scaler.transform(x)

    p_mlp = float(mlp.predict(x_s)[0])
    p_ridge = float(ridge.predict(x_s)[0])
    p_en = p_mlp * w_mlp_n + p_ridge * w_ridge_n

    meta = raw.get('META', {}) if is_rpe else {}
    label = f' ({meta.get("level","")})' if is_rpe else ''
    print(f'\n  {name}{label}:')
    print(f'    MLP={p_mlp:.2f}  Ridge={p_ridge:.2f}  集成={p_en:.2f}')
    if feats:
        for k in ['hand_speed_index', 'tap_per_second', 'notes_per_second',
                   'max_simultaneous', 'tap_burst_top5', 'sustained_density_run_count']:
            print(f'    {k}: {feats.get(k,0):.2f}')

# =======================================
# 保存
# =======================================
print('\n--- 保存模型 ---')
model_data = {
    'mlp': mlp, 'ridge': ridge, 'scaler': scaler,
    'feature_names': feature_names,
    'weight_mlp': w_mlp_n, 'weight_ridge': w_ridge_n,
    'metrics': {
        'mlp_r2': r2_mlp, 'mlp_mae': mae_mlp,
        'ridge_r2': r2_ridge, 'ridge_mae': mae_ridge,
        'ensemble_r2': r2_en, 'ensemble_mae': mae_en,
    }
}
save_path = os.path.join(os.path.dirname(__file__), 'models', 'nn_model.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f'  已保存到: {save_path}')

print(f'\n  指标总览:')
for k, v in model_data['metrics'].items():
    print(f'    {k}: {v:.4f}')
print(f'  特征数: {len(feature_names)}')
