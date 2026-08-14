"""
Phigros 难度预测系统 v7 — 维度贡献数学均衡版

改进点:
  1. 配置权重 ×0.48  (总co从2.31→1.11, 占比从57%→28%)
  2. 读谱权重 ×3.50  (总co从0.16→0.56, 占比从4%→15%)
  3. 密度权重 ×1.30  (总co从0.52→0.68, 占比从13%→19%)
  4. 位移权重 ×1.20  (总co从0.50→0.60, 占比从12%→16%)
  5. dynamic_cap 微调适配新分布 (knee:1.2→1.0, power:0.95→0.90)
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
import math

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
sys.path.insert(0, os.path.dirname(__file__))

print('='*70)
print('  Phigros 难度预测系统 v7 — 维度贡献均衡版')
print('  训练集: 官谱957 (无自定义谱)')
print('  配置×0.48 | 读谱×3.50 | 密度×1.30 | 位移×1.20 | 耐力×1.00')
print('='*70)

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)
all_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    for lv in ['EZ','HD','IN','AT']:
        if lv in info['levels'] and lv in diffs:
            all_items.append({'folder':fn,'filepath':info['levels'][lv],'difficulty':diffs[lv],'level':lv})
print(f'\n官方谱面: {len(all_items)}')

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
    except: pass
    if (i+1)%300==0: print(f'  加载 {i+1}/{len(all_items)}')

print(f'  官方提取: {len(feats_list)}')

feature_names = sorted(feats_list[0].keys())
n_samples = len(feats_list)

# ====== GB特征过滤 ======
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

X_full = np.array([[f.get(n,0) for n in feature_names] for f in feats_list])
X_full_gb = np.array([[f.get(n,0) for n in gb_feature_names] for f in feats_list])
y_full = np.array(labels)

print(f'\n总谱面: {n_samples}, GB特征: {len(gb_feature_names)}, 难度: {y_full.min():.1f}~{y_full.max():.1f}')

# P95/P99只用官方数据
p95_vals, p99_vals = {}, {}
for j, name in enumerate(feature_names):
    col = np.array([f.get(name,0) for f in feats_list])
    p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
    p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0

# ====== 维度均衡后的 FLAT_FEATURES ======
# 数学优化：配置×0.48 | 读谱×3.50 | 密度×1.30 | 位移×1.20 | 耐力×1.00
# 目标：配置28% 耐力21% 密度19% 位移16% 读谱15%
FLAT_FEATURES = [
    # === 密度 (×1.30) ===
    ('density_dimension', 1.0, 0.546),
    ('core_peak_density_1sec_top5avg', 8, 0.065),
    ('core_peak_density_top5avg_1beat', 0.5, 0.065),

    # === 位移 (×1.20) ===
    ('movement_per_second', 3.0, 0.264),
    ('burst_avg_movement', 0.5, 0.120),
    ('wide_jump_density', 0.5, 0.120),
    ('sim_pos_spread_max', 3, 0.096),

    # === 配置 (×0.48 — 从2.31降至1.11) ===
    ('stair_density', 1.0, 0.0864),
    ('stair_speed_avg', 8.0, 0.072),
    ('stair_complexity', 0.2, 0.048),
    ('stair_chord_ratio', 0.3, 0.0384),
    ('trill_density', 2.0, 0.048),
    ('jack_density', 2.0, 0.0576),
    ('chord_size_entropy', 0.5, 0.120),
    ('sim_pos_spread_mean', 1.0, 0.048),
    ('multi_finger_3plus_events', 10, 0.024),
    ('chord_alternation_rate', 0.5, 0.0816),
    ('weighted_mf_score_per_sec', 10, 0.0816),
    ('discrete_mf_ratio', 0.3, 0.0576),
    ('position_cluster_count', 4, 0.0576),
    ('track_deviation_score', 0.3, 0.0384),
    ('position_entropy', 2.0, 0.048),
    ('position_range_used', 0.5, 0.0288),
    ('pattern_switch_rate', 1.0, 0.048),
    ('direction_irregularity', 0.5, 0.0384),
    ('hold_interference_index', 0.3, 0.048),
    ('drag_flick_ratio', 0.3, 0.0384),

    # === 耐力 (×1.00 — 不变) ===
    ('stamina_ratio', 0.3, 0.15),
    ('tap_per_second', 2.5, 0.12),
    ('total_notes', 400, 0.06),
    ('tap_count', 400, 0.06),
    ('duration_sec', 100, 0.06),
    ('rest_ratio', 0.3, 0.06),
    ('global_jack_count', 20, 0.06),
    ('burst_intensity_mean', 0.3, 0.08),
    ('tap_burst_top5', 0.5, 0.08),

    # === 读谱 (×3.50 — 从0.16跃至0.56) ===
    ('density_transition_mean', 0.15, 0.28),
    ('density_transition_std', 0.2, 0.28),
    ('tempo_change_count', 50, 0.28),
    ('offbeat_ratio', 0.04, 0.28),
    ('rhythm_entropy', 2.5, 0.21),
    ('type_switch_per_sec', 0.4, 0.21),
    ('note_clutter_ratio', 0.05, 0.21),
]

# dynamic_cap: 降低knee适配新分布（总boost变低）
DC = {'knee': 1.0, 'power': 0.90}

def _compute_dim_boost(feats, p95, p99, feat_list):
    raw = 0.0
    for fname, baseline, coeff in feat_list:
        val = feats.get(fname, 0)
        pv = p95.get(fname, 0)
        thresh = max(pv * 0.55, baseline * 0.5)
        if val <= thresh:
            continue
        excess = val / thresh - 1.0
        contrib = coeff * (excess ** 0.70)
        if val > max(p99.get(fname, 0), baseline * 0.5):
            p99_excess = val / max(p99.get(fname, 0), baseline * 0.5) - 1.0
            p99_bonus = coeff * max(0, p99_excess) ** 0.70 * 0.5
            contrib += p99_bonus
        raw += contrib
    return raw

def _dynamic_cap(raw):
    KNEE = DC['knee']; POWER = DC['power']
    if raw <= KNEE:
        return raw
    excess = raw - KNEE
    return KNEE + excess ** POWER

def compute_simple_boost(feats, p95, p99):
    total_boost = _compute_dim_boost(feats, p95, p99, FLAT_FEATURES)
    total_boost = _dynamic_cap(total_boost)
    return total_boost, {'total_boost': round(total_boost, 4)}

# ====== 联合训练 ======
print('\n--- 联合训练 GB+boost (v7) ---')
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
bins = np.digitize(y_full, bins=[0,5,7,9,11,13,14,15,16,16.5,17,18])
train_idx, test_idx = next(sss.split(X_full_gb, bins))

scaler_gb = StandardScaler()
X_tr_s = scaler_gb.fit_transform(X_full_gb[train_idx])
X_te_s = scaler_gb.transform(X_full_gb[test_idx])

y_tr_labels = y_full[train_idx].copy()
y_te_labels = y_full[test_idx].copy()
y_te_orig_labels = y_te_labels.copy()

print('  计算boost...')
all_boosts = np.array([compute_simple_boost(feats_list[i], p95_vals, p99_vals)[0] for i in range(n_samples)])
train_boosts = all_boosts[train_idx]
test_boosts = all_boosts[test_idx]

y_tr_residual = y_tr_labels - train_boosts
y_te_residual = y_te_labels - test_boosts

print(f'  Boost范围: [{all_boosts.min():.3f}, {all_boosts.max():.3f}]')
print(f'  训练集残差范围: [{y_tr_residual.min():.2f}, {y_tr_residual.max():.2f}]')

gb = GradientBoostingRegressor(n_estimators=700, max_depth=5, min_samples_leaf=3,
                                learning_rate=0.05, subsample=0.8, random_state=42)
gb.fit(X_tr_s, y_tr_residual)

y_pred_gb = gb.predict(X_te_s)
y_pred_final = y_pred_gb + test_boosts

r2 = r2_score(y_te_orig_labels, y_pred_final)
mae = mean_absolute_error(y_te_orig_labels, y_pred_final)
print(f'  测试集: R2={r2:.4f}, MAE={mae:.4f}')

# 全量训练
X_all_s = scaler_gb.fit_transform(X_full_gb)
y_all_residual = y_full - all_boosts
gb_full = GradientBoostingRegressor(n_estimators=700, max_depth=5, min_samples_leaf=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
gb_full.fit(X_all_s, y_all_residual)
print(f'  全量训练完成 (n={n_samples})')

# ====== 分档统计 ======
print('\n' + '='*70)
print('  训练集内评估 + 分档统计')
print('='*70)

BINS = np.array([0,5,7,9,11,12,13,14,15,16,17,18,20])
n_bins = len(BINS) - 1
boost_per_bin = [[] for _ in range(n_bins)]

for i in range(n_samples):
    x = np.array([[feats_list[i].get(n,0) for n in gb_feature_names]])
    xs = scaler_gb.transform(x)
    p_gb = float(gb_full.predict(xs)[0])
    p_b, _ = compute_simple_boost(feats_list[i], p95_vals, p99_vals)
    p_f = p_gb + p_b
    if i < 5 or i > n_samples-5:
        print(f'{names_list[i]:<35} 真={labels[i]:.1f}  GB={p_gb:.3f}  +Boost={p_b:.3f}  ={p_f:.3f}  [{p_f-labels[i]:+.3f}]')
    
    for j in range(n_bins):
        if BINS[j] <= p_f < BINS[j+1]:
            boost_per_bin[j].append(float(p_b))
            break

if len(all_items) < n_samples:
    print(f'  ... (显示前5和后5个, 共{n_samples})')

boost_bin_stats = {}
for j in range(n_bins):
    arr = np.array(boost_per_bin[j]) if boost_per_bin[j] else np.array([0])
    boost_bin_stats[f'{BINS[j]:.0f}-{BINS[j+1]:.0f}'] = {
        'median': float(np.median(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
        'count': len(arr),
    }

print(f'\n  Boost分档:')
for k, v in boost_bin_stats.items():
    iqr = v['q75'] - v['q25']
    print(f'    [{k}): median={v["median"]:.2f}  IQR={iqr:.2f}  n={v["count"]}')

# ====== 保存模型 ======
model_out = {
    'gb': gb_full, 'scaler': scaler_gb, 'feature_names': gb_feature_names,
    'p95_vals': p95_vals, 'p99_vals': p99_vals,
    'FLAT_FEATURES': FLAT_FEATURES,
    'dynamic_cap': DC,
    'boost_bin_stats': boost_bin_stats,
    'dimension_factors': {
        '密度': 1.30, '平均位移': 1.20, '配置': 0.48, '耐力': 1.00, '读谱': 3.50
    },
    'metrics': {'r2': r2, 'mae': mae, 'n_train': n_samples},
}
out_path = os.path.join(os.path.dirname(__file__), 'models', '6dim_model_v7.pkl')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'wb') as f:
    pickle.dump(model_out, f)
print(f'\n  模型已保存: {out_path}')

# ====== 维度占比验证 ======
print('\n--- 维度均衡验证 (官谱平均) ---')
cat_def = {
    '密度': ['density_dimension', 'core_peak_density_1sec_top5avg', 'core_peak_density_top5avg_1beat'],
    '平均位移': ['movement_per_second', 'burst_avg_movement', 'wide_jump_density', 'sim_pos_spread_max'],
    '配置': ['stair_density', 'stair_speed_avg', 'stair_complexity', 'stair_chord_ratio', 'trill_density',
             'jack_density', 'chord_size_entropy', 'sim_pos_spread_mean', 'multi_finger_3plus_events',
             'weighted_mf_score_per_sec', 'discrete_mf_ratio', 'chord_alternation_rate',
             'position_cluster_count', 'track_deviation_score', 'position_entropy', 'position_range_used',
             'pattern_switch_rate', 'direction_irregularity', 'hold_interference_index', 'drag_flick_ratio'],
    '耐力': ['stamina_ratio', 'tap_per_second', 'total_notes', 'tap_count', 'duration_sec',
             'rest_ratio', 'global_jack_count', 'burst_intensity_mean', 'tap_burst_top5'],
    '读谱': ['density_transition_mean', 'density_transition_std', 'tempo_change_count', 'offbeat_ratio',
             'rhythm_entropy', 'type_switch_per_sec', 'note_clutter_ratio'],
}
cat_contribs = {k: [] for k in cat_def}
for i in range(min(200, n_samples)):
    feats = feats_list[i]
    for cat_name, cat_feats in cat_def.items():
        contrib = 0
        for fname, bl, co in FLAT_FEATURES:
            if fname in cat_feats:
                val = feats.get(fname, 0)
                pv = p95_vals.get(fname, 0)
                thresh = max(pv * 0.55, bl * 0.5)
                if val > thresh:
                    excess = val / thresh - 1.0
                    contrib += co * (excess ** 0.70)
                    if val > max(p99_vals.get(fname, 0), bl * 0.5):
                        p99_excess = val / max(p99_vals.get(fname, 0), bl * 0.5) - 1.0
                        contrib += co * max(0, p99_excess) ** 0.70 * 0.5
        cat_contribs[cat_name].append(contrib)

print(f"{'维度':<10} {'平均贡献':>10} {'占比%':>8}")
total_avg = sum(np.mean(cat_contribs[k]) for k in cat_def)
for k in ['密度', '平均位移', '配置', '耐力', '读谱']:
    avg = np.mean(cat_contribs[k])
    pct = avg/total_avg*100 if total_avg > 0 else 0
    print(f"{k:<10} {avg:>10.4f} {pct:>7.1f}%")

# 保存CSV
import csv
csv_path = out_path.replace('.pkl', '_predictions.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['谱面', '难度', 'GB', 'Boost', '预测', '误差'])
    for i in range(n_samples):
        x = np.array([[feats_list[i].get(n,0) for n in gb_feature_names]])
        xs = scaler_gb.transform(x)
        p_gb = float(gb_full.predict(xs)[0])
        p_b, _ = compute_simple_boost(feats_list[i], p95_vals, p99_vals)
        p_f = p_gb + p_b
        w.writerow([names_list[i], labels[i], round(p_gb,3), round(p_b,3), round(p_f,3), round(p_f-labels[i],3)])
print(f'\n  预测CSV已保存: {csv_path}')
print('\n' + '='*70)
print('  v7训练完成!')
print('='*70)
