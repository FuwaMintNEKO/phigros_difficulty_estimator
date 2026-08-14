"""
全官方谱面重训 v9 (随机5%测试集)

与 app.py 推理完全对齐:
  - boost 使用 app.py 的 v9 MANUAL_FLAT 配置 (不再用 pkl 内置的 v7 FLAT_FEATURES)
  - P95/P99 阈值只从训练折计算
  - GB 目标 = 真定数 - v9 boost
  - 随机 5% 测试集 (seed=42), 训练/测试不重叠

特征提取器已修复: real_active 气泡并集法 + speedEvents 兼容 RPE eventLayers
"""
import os, sys, json, pickle, numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from boost_config import MANUAL_FLAT  # 与推理共享同一份 boost 配置

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
OUT_PATH = os.path.join(_ROOT, 'models', '6dim_model_v7_retrain.pkl')

print('='*70)
print('  全官方谱面重训 v9 (随机5%测试集, seed=42)')
print('='*70)

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)
all_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties:
        continue
    diffs = song_difficulties[sid]
    for lv in ['EZ','HD','IN','AT']:
        if lv in info['levels'] and lv in diffs:
            all_items.append({'folder': fn, 'filepath': info['levels'][lv],
                              'difficulty': diffs[lv], 'level': lv})
print(f'官谱总数: {len(all_items)}')

feats_list, labels, levels_list, names_list = [], [], [], []
for i, item in enumerate(all_items):
    try:
        cd = load_chart_json(item['filepath'])
        feats = extract_features(cd)
        if feats:
            feats_list.append(feats)
            labels.append(item['difficulty'])
            levels_list.append(item['level'])
            names_list.append(item['folder'])
    except Exception as e:
        print(f'  跳过 {item["folder"]}/{item["level"]}: {e}')
    if (i+1) % 300 == 0:
        print(f'  加载 {i+1}/{len(all_items)}')
print(f'成功提取特征: {len(feats_list)}')

feature_names = sorted(feats_list[0].keys())
n_samples = len(feats_list)

# ====== GB特征过滤 (沿用 v7 规则) ======
GB_EXCLUDE_KEYWORDS = [
    'stop_go', 'track_section', 'offbeat_ratio', 'dense_mf',
    'mf_burst', 'mf_events_per_second', 'mf_with_hold',
    'cross_line_3plus', 'min_interval_beats',
    'multi_finger_3plus', 'multi_finger_4plus', 'multi_finger_max',
    'chord_size_entropy', 'chord_3note', 'chord_4plus',
    'long_jack', 'short_jack', 'jack_max_run',
    'per_second', 'per_sec', 'rate_per_sec',
    'total_movement', 'total_steps', 'total_event',
    'total_hold_duration', 'total_chord',
    'speed_change_total',
    'micro_max_', 'micro_spike_',
    'density_above_zero', 'core_density_above_zero',
    'density_skew', 'density_transition_max',
    'avg_hold_duration', 'max_hold_duration',
    'finger_vs_total',
]
GB_KEEP = {
    'density_dimension',
    'real_core_notes_per_second',
    'core_peak_density_1sec_top5avg',
    'core_peak_density_top5avg_1beat',
}
gb_feature_names = [n for n in feature_names
                    if n in GB_KEEP or not any(kw in n for kw in GB_EXCLUDE_KEYWORDS)]
print(f'GB特征数: {len(gb_feature_names)}')

X_full = np.array([[f.get(n, 0) for n in gb_feature_names] for f in feats_list])
y_full = np.array(labels)

# ====== 随机5%测试集 (seed=42) ======
idx = np.arange(n_samples)
train_idx, test_idx = train_test_split(idx, test_size=0.05, random_state=42)
train_idx = np.sort(train_idx)
test_idx = np.sort(test_idx)
print(f'训练集: {len(train_idx)}, 测试集: {len(test_idx)}')

# 测试集难度构成
from collections import Counter
test_levels = Counter(levels_list[i] for i in test_idx)
print(f'测试集难度构成: {dict(test_levels)}')

# ====== P95/P99 只从训练折计算 ======
p95_vals, p99_vals = {}, {}
for j, name in enumerate(gb_feature_names):
    col = X_full[train_idx, j]
    p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
    p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0

# ====== v9 boost (与 app.py compute_boost 一致, speed=1, excess_exp=0.70) ======
def compute_boost_v9(feats):
    total = 0.0
    for fname, bl, co in MANUAL_FLAT:
        v = feats.get(fname, 0)
        pv = p95_vals.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t:
            continue
        e = v / t - 1.0
        x = co * (e ** 0.70)
        if v > max(p99_vals.get(fname, 0), bl * 0.5):
            pe = v / max(p99_vals.get(fname, 0), bl * 0.5) - 1.0
            x += co * max(0, pe) ** 0.70 * 0.5
        total += x
    return total

print('计算 v9 boost...')
all_boosts = np.array([compute_boost_v9(f) for f in feats_list])
print(f'Boost范围: [{all_boosts.min():.3f}, {all_boosts.max():.3f}]')

train_boosts = all_boosts[train_idx]
test_boosts = all_boosts[test_idx]

y_tr_residual = y_full[train_idx] - train_boosts
y_te_residual = y_full[test_idx] - test_boosts
print(f'训练残差范围: [{y_tr_residual.min():.2f}, {y_tr_residual.max():.2f}]')

# ====== GB 训练 (沿用 v7 超参) ======
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_full[train_idx])
X_te_s = scaler.transform(X_full[test_idx])

gb = GradientBoostingRegressor(n_estimators=700, max_depth=5, min_samples_leaf=3,
                               learning_rate=0.05, subsample=0.8, random_state=42)
gb.fit(X_tr_s, y_tr_residual)
print('GB 训练完成')

# ====== 评估 ======
def evaluate(Xs, y_true, boosts, label):
    p_gb = gb.predict(Xs)
    p_final = p_gb + boosts
    r2 = r2_score(y_true, p_final)
    mae = mean_absolute_error(y_true, p_final)
    bias = float(np.mean(p_final - y_true))
    print(f'\n[{label}] n={len(y_true)}  R2={r2:.4f}  MAE={mae:.4f}  平均偏差={bias:+.3f}')
    return p_final, p_gb

print('\n' + '='*70)
p_tr_final, p_tr_gb = evaluate(X_tr_s, y_full[train_idx], train_boosts, '训练集')
p_te_final, p_te_gb = evaluate(X_te_s, y_full[test_idx], test_boosts, '测试集(5%)')

# 测试集分档偏差
print('\n=== 测试集分档偏差 (按真定数) ===')
BANDS = [('<4', 0, 4), ('4-7', 4, 7), ('7-11', 7, 11), ('11-14', 11, 14), ('14-16.5', 14, 16.5), ('>16.5', 16.5, 99)]
for name, lo, hi in BANDS:
    mask = (y_full[test_idx] >= lo) & (y_full[test_idx] < hi)
    if np.sum(mask) == 0:
        continue
    errs = p_te_final[mask] - y_full[test_idx][mask]
    gb_vals = p_te_gb[mask]
    b_vals = test_boosts[mask]
    print(f'  [{name}]: n={np.sum(mask):<3} 偏差={np.mean(errs):+.2f} '
          f'GB均值={np.mean(gb_vals):.2f} boost均值={np.mean(b_vals):.2f}')

# 测试集 Top 误差
print('\n=== 测试集误差最大 Top8 ===')
te_sorted = sorted(zip(test_idx, p_te_final, p_te_gb, test_boosts),
                   key=lambda x: -abs(x[1] - y_full[x[0]]))
for ti, pf, pg, pb in te_sorted[:8]:
    print(f'  {names_list[ti]:<32} {levels_list[ti]:<3} 真={y_full[ti]:.1f} '
          f'GB={pg:.2f} +boost={pb:.2f} ={pf:.2f} 偏差={pf-y_full[ti]:+.2f}')

# ====== 保存模型 ======
model_out = {
    'gb': gb, 'scaler': scaler, 'feature_names': gb_feature_names,
    'p95_vals': p95_vals, 'p99_vals': p99_vals,
    'FLAT_FEATURES': MANUAL_FLAT,
    'dynamic_cap': None,
    'metrics': {
        'r2_train': float(r2_score(y_full[train_idx], p_tr_final)),
        'mae_train': float(mean_absolute_error(y_full[train_idx], p_tr_final)),
        'r2_test': float(r2_score(y_full[test_idx], p_te_final)),
        'mae_test': float(mean_absolute_error(y_full[test_idx], p_te_final)),
        'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)),
        'test_levels': dict(test_levels),
        'random_state': 42, 'test_size': 0.05,
        'note': 'boost 配置与 app.py v9 MANUAL_FLAT 对齐; P95/P99 仅来自训练折',
    },
}
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, 'wb') as f:
    pickle.dump(model_out, f)
print(f'\n模型已保存: {OUT_PATH}')
