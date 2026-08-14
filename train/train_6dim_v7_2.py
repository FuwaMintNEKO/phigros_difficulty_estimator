"""
Phigros v7.2 — BPM修复版
  变更：
  1. collect_all_notes: 无BPMList时每条线用自己的BPM，不再统一用第一条线
  2. predict_rpe.py: 转换时保留BPMList（修正RPE变速谱）
  3. _parse_bpm_timeline: 兼容float格式startTime
  4. 维度因子沿用v7.1：配置×0.55 | 读谱×2.00 | 密度×1.20 | 位移×1.10 | 耐力×1.00
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

FACTORS = {'密度': 1.20, '平均位移': 1.10, '配置': 0.55, '耐力': 1.00, '读谱': 2.00}

print('='*70)
print('  Phigros 难度预测系统 v7.2 — BPM修复版')
print(f'  配置×{FACTORS["配置"]} | 读谱×{FACTORS["读谱"]} | 密度×{FACTORS["密度"]} | 位移×{FACTORS["平均位移"]} | 耐力×{FACTORS["耐力"]}')
print('  [新] per-line BPM | RPE变速保留 | startTime兼容')
print('='*70)

# ====== 加载官谱训练数据 ======
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

# GB特征过滤（同v6/v7.1）
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
X_full_gb = np.array([[f.get(n,0) for n in gb_feature_names] for f in feats_list])
y_full = np.array(labels)
print(f'\n总谱面: {n_samples}, GB特征: {len(gb_feature_names)}, 难度: {y_full.min():.1f}~{y_full.max():.1f}')

# P95/P99（从修正后的特征重新计算）
p95_vals, p99_vals = {}, {}
for j, name in enumerate(feature_names):
    col = np.array([f.get(name,0) for f in feats_list])
    p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
    p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0

# 维度均衡后的 FLAT_FEATURES（沿用v7.1的co值）
FLAT_FEATURES = [
    # === 密度 (co_sum=0.624, ×1.20) ===
    ('density_dimension', 1.0, 0.504),
    ('core_peak_density_1sec_top5avg', 8, 0.060),
    ('core_peak_density_top5avg_1beat', 0.5, 0.060),
    # === 位移 (co_sum=0.550, ×1.10) ===
    ('movement_per_second', 3.0, 0.242),
    ('burst_avg_movement', 0.5, 0.110),
    ('wide_jump_density', 0.5, 0.110),
    ('sim_pos_spread_max', 3, 0.088),
    # === 配置 (co_sum=1.271, ×0.55) ===
    ('stair_density', 1.0, 0.099),
    ('stair_speed_avg', 8.0, 0.0825),
    ('stair_complexity', 0.2, 0.055),
    ('stair_chord_ratio', 0.3, 0.044),
    ('trill_density', 2.0, 0.055),
    ('jack_density', 2.0, 0.066),
    ('chord_size_entropy', 0.5, 0.1375),
    ('sim_pos_spread_mean', 1.0, 0.055),
    ('multi_finger_3plus_events', 10, 0.0275),
    ('chord_alternation_rate', 0.5, 0.0935),
    ('weighted_mf_score_per_sec', 10, 0.0935),
    ('discrete_mf_ratio', 0.3, 0.066),
    ('position_cluster_count', 4, 0.066),
    ('track_deviation_score', 0.3, 0.044),
    ('position_entropy', 2.0, 0.055),
    ('position_range_used', 0.5, 0.033),
    ('pattern_switch_rate', 1.0, 0.055),
    ('direction_irregularity', 0.5, 0.044),
    ('hold_interference_index', 0.3, 0.055),
    ('drag_flick_ratio', 0.3, 0.044),
    # === 耐力 (co_sum=0.76, ×1.00) ===
    ('stamina_ratio', 0.3, 0.15),
    ('tap_per_second', 2.5, 0.12),
    ('total_notes', 400, 0.06),
    ('tap_count', 400, 0.06),
    ('duration_sec', 100, 0.06),
    ('rest_ratio', 0.3, 0.06),
    ('global_jack_count', 20, 0.06),
    ('burst_intensity_mean', 0.3, 0.08),
    ('tap_burst_top5', 0.5, 0.08),
    # === 读谱 (co_sum=0.84, ×2.00) ===
    ('density_transition_mean', 0.15, 0.16),
    ('density_transition_std', 0.2, 0.16),
    ('tempo_change_count', 50, 0.16),
    ('offbeat_ratio', 0.04, 0.16),
    ('rhythm_entropy', 2.5, 0.12),
    ('type_switch_per_sec', 0.4, 0.12),
    ('note_clutter_ratio', 0.05, 0.12),
]

DC = {'knee': 1.0, 'power': 0.90}

def _compute_dim_boost(feats, p95, p99, feat_list):
    raw = 0.0
    for fname, baseline, coeff in feat_list:
        val = feats.get(fname, 0)
        pv = p95.get(fname, 0)
        thresh = max(pv * 0.55, baseline * 0.5)
        if val <= thresh: continue
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
    if raw <= KNEE: return raw
    return KNEE + (raw - KNEE) ** POWER

def compute_simple_boost(feats, p95, p99):
    total_boost = _compute_dim_boost(feats, p95, p99, FLAT_FEATURES)
    total_boost = _dynamic_cap(total_boost)
    return total_boost, {'total_boost': round(total_boost, 4)}

# ====== 联合训练 GB+boost ======
print('\n--- 联合训练 GB+boost (v7.2) ---')
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
bins = np.digitize(y_full, bins=[0,5,7,9,11,13,14,15,16,16.5,17,18])
train_idx, test_idx = next(sss.split(X_full_gb, bins))

scaler_gb = StandardScaler()
X_tr_s = scaler_gb.fit_transform(X_full_gb[train_idx])
X_te_s = scaler_gb.transform(X_full_gb[test_idx])
y_tr_labels, y_te_labels = y_full[train_idx].copy(), y_full[test_idx].copy()
y_te_orig_labels = y_te_labels.copy()

print('  计算boost...')
all_boosts = np.array([compute_simple_boost(feats_list[i], p95_vals, p99_vals)[0] for i in range(n_samples)])
train_boosts, test_boosts = all_boosts[train_idx], all_boosts[test_idx]
y_tr_residual = y_tr_labels - train_boosts
y_te_residual = y_te_labels - test_boosts
print(f'  Boost范围: [{all_boosts.min():.3f}, {all_boosts.max():.3f}]')

gb = GradientBoostingRegressor(n_estimators=700, max_depth=5, min_samples_leaf=3,
                                learning_rate=0.05, subsample=0.8, random_state=42)
gb.fit(X_tr_s, y_tr_residual)
y_pred_final = gb.predict(X_te_s) + test_boosts
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
BINS = np.array([0,5,7,9,11,12,13,14,15,16,17,18,20])
n_bins = len(BINS) - 1
boost_per_bin = [[] for _ in range(n_bins)]
for i in range(n_samples):
    x = np.array([[feats_list[i].get(n,0) for n in gb_feature_names]])
    xs = scaler_gb.transform(x)
    p_gb = float(gb_full.predict(xs)[0])
    p_b, _ = compute_simple_boost(feats_list[i], p95_vals, p99_vals)
    p_f = p_gb + p_b
    for j in range(n_bins):
        if BINS[j] <= p_f < BINS[j+1]:
            boost_per_bin[j].append(float(p_b))
            break

boost_bin_stats = {}
for j in range(n_bins):
    arr = np.array(boost_per_bin[j]) if boost_per_bin[j] else np.array([0])
    boost_bin_stats[f'{BINS[j]:.0f}-{BINS[j+1]:.0f}'] = {
        'median': float(np.median(arr)), 'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)), 'count': len(arr),
    }

# 保存模型
model_out = {
    'gb': gb_full, 'scaler': scaler_gb, 'feature_names': gb_feature_names,
    'p95_vals': p95_vals, 'p99_vals': p99_vals,
    'FLAT_FEATURES': FLAT_FEATURES,
    'dynamic_cap': DC,
    'boost_bin_stats': boost_bin_stats,
    'dimension_factors': FACTORS,
    'metrics': {'r2': r2, 'mae': mae, 'n_train': n_samples},
}
out_path = os.path.join(os.path.dirname(__file__), 'models', '6dim_model_v7_2.pkl')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'wb') as f:
    pickle.dump(model_out, f)
print(f'\n  模型已保存: {out_path}')

# ====== 维度占比验证 ======
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
for i in range(min(n_samples, 300)):
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

print(f"\n维度均衡验证 (官谱前300):")
total_avg = sum(np.mean(cat_contribs[k]) for k in cat_def)
for k in ['密度', '平均位移', '配置', '耐力', '读谱']:
    avg = np.mean(cat_contribs[k])
    pct = avg/total_avg*100 if total_avg > 0 else 0
    print(f"  {k}: {avg:.4f} ({pct:.1f}%)")

# ====== 导出预测CSV ======
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
print(f'\n预测CSV: {csv_path}')

# ====== 关键特征P95对比（v7.1 vs v7.2） ======
old_model_path = os.path.join(os.path.dirname(__file__), 'models', '6dim_model_v7_1.pkl')
if os.path.exists(old_model_path):
    with open(old_model_path, 'rb') as f:
        old_m = pickle.load(f)
    old_p95 = old_m.get('p95_vals', {})
    
    print(f'\n{"特征":<40} {"v7.1 P95":>10} {"v7.2 P95":>10} {"变化":>8}')
    print('-' * 70)
    key_feats = ['density_dimension', 'tap_per_second', 'duration_sec', 'movement_per_second',
                 'core_peak_density_1sec_top5avg', 'stair_density', 'stamina_ratio']
    for kf in key_feats:
        o = old_p95.get(kf, 0)
        n = p95_vals.get(kf, 0)
        delta = (n - o) / max(o, 0.001) * 100 if o > 0 else 0
        mark = ' ***' if abs(delta) > 2 else ''
        print(f'  {kf:<40} {o:>10.3f} {n:>10.3f} {delta:>+7.1f}%{mark}')

print('='*70)
print('  v7.2训练完成!')
print('='*70)
