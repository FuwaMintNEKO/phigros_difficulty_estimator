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

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
sys.path.insert(0, os.path.dirname(__file__))
from unified_parser import load_chart

print('='*70)
print('  Phigros 难度预测系统 v6（5维：密度/位移/配置/耐力/读谱）')
print('  训练集: 官谱957 (无自定义谱)')
print('  密度维度统一化: density_dimension = √(总真实TPS × 1s峰值TPS)')
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

DOWNLOADS = r'C:\Users\NaNK\Downloads'

# 训练集仅使用官谱，无自定义谱
custom_charts = []

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

for name, path_suffix, diff, lv in custom_charts:
    fp = os.path.join(DOWNLOADS, path_suffix)
    try:
        cd = load_chart(fp)
        feats = extract_features(cd)
        if feats:
            feats_list.append(feats)
            labels.append(diff)
            levels_list.append(lv)
            names_list.append(f'{name}(自制)')
            print(f'  加入: {name} = {diff}')
    except Exception as e:
        print(f'  {name}失败: {e}')

feature_names = sorted(feats_list[0].keys())
n_samples = len(feats_list)

# ====== GB特征过滤：只保留有区分力的特征 ======
# 原则：GB应该学"谱面结构"而非"多少量"
# 剔除：1) p95≈0的稀疏特征  2) per_second/per_sec速率  3) total_绝对值  4) 微爆发常数
GB_EXCLUDE_KEYWORDS = [
    # --- p95≈0: 训练集几乎不出现，GB完全学不到 ---
    'stop_go', 'track_section', 'offbeat_ratio', 'dense_mf',
    'mf_burst', 'mf_events_per_second', 'mf_with_hold',
    'cross_line_3plus', 'min_interval_beats',
    'multi_finger_3plus', 'multi_finger_4plus', 'multi_finger_max',
    'chord_size_entropy', 'chord_3note', 'chord_4plus',
    'long_jack', 'short_jack', 'jack_max_run',
    # --- per_second/per_sec 速率（时间归一化，受BPM/时长强烈影响）---
    'per_second', 'per_sec', 'rate_per_sec',  # 匹配所有含这些子串的特征
    # --- total_ 绝对值（随谱面长度线性增长）---
    'total_movement', 'total_steps', 'total_event',
    'total_hold_duration', 'total_chord',
    'speed_change_total',
    # --- 微爆发（几乎常数，CoV<1）---
    'micro_max_', 'micro_spike_',
    # --- 密度窗口统计（保留density_dimension即可）---
    'density_above_zero', 'core_density_above_zero',
    'density_skew', 'density_transition_max',
    # --- hold时长统计（绝对量）---
    'avg_hold_duration', 'max_hold_duration',
    # --- 手指比例（几乎常数）---
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

print(f'\n总谱面: {n_samples}, 全特征: {len(feature_names)}, GB特征: {len(gb_feature_names)}, 难度: {y_full.min():.1f}~{y_full.max():.1f}')
print(f'  GB剔除特征数: {len(feature_names) - len(gb_feature_names)}')

# P95/P99只用官方数据
official_n = len(all_items)
official_feats = feats_list[:official_n]
p95_vals, p99_vals = {}, {}
for j, name in enumerate(feature_names):
    col = np.array([f.get(name,0) for f in official_feats])
    p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
    p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0

# ====== Boost设计 ======
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

# ====== 平铺特征定义（5维：密度/位移/配置/耐力/读谱） ======
# 注：密度+1smax已合并为density_dimension为核心，2个细化特征为辅
FLAT_FEATURES = [
    # === 密度 (Density — 统一维度: √(总真实核心TPS × 1s峰值核心密度)) ===
    # 几何平均确保持续密度和爆发密度都高时难度才高，一方低则综合分低
    ('density_dimension', 1.0, 0.42),
    # 辅助细化：极高峰值 + 极细粒度尖峰（系数极低，仅作微调）
    ('core_peak_density_1sec_top5avg', 8, 0.05),
    ('core_peak_density_top5avg_1beat', 0.5, 0.05),

    # === 位移 (Movement) ===
    ('movement_per_second', 3.0, 0.22),  # 每秒位移量
    ('burst_avg_movement', 0.5, 0.10),  # 爆发段平均位移
    ('wide_jump_density', 0.5, 0.10),  # 大跳密度
    ('sim_pos_spread_max', 3, 0.08),  # 最大同时音符跨度

    # === 配置 (Configuration — 谱面排列方式复杂度) ===
    # 楼梯（v2和弦感知，多指谱核心特征——4~6k楼梯/爬升/山峰型）
    ('stair_density', 1.0, 0.18),          # 楼梯速率（步/秒，和弦分组后检测）
    ('stair_speed_avg', 8.0, 0.15),        # 楼梯平均速度（步/秒，越快越难）
    ('stair_complexity', 0.2, 0.10),       # 楼梯复杂度（方向变化占比，山峰型>单方向）
    ('stair_chord_ratio', 0.3, 0.08),      # 和弦楼梯占比（和弦参与的楼梯比例）
    # 颤音 / 纵连
    ('trill_density', 2.0, 0.10),          # 颤音密度（连续左右交替）
    ('jack_density', 2.0, 0.12),           # 纵连密度（同位置连续击打）
    # 和弦配置（多指谱核心——区分多指vs双指）
    ('chord_size_entropy', 0.5, 0.25),     # 和弦大小熵（和弦大小变化复杂度）
    ('sim_pos_spread_mean', 1.0, 0.10),    # 和弦跨度均值（同时音符的伸展度）
    ('multi_finger_3plus_events', 10, 0.05),  # 多指事件数（≥3指同时，非配置专属）
    ('chord_alternation_rate', 0.5, 0.17), # 和弦交替率（和弦↔单点交替频率）
    # 加权多指协调（区分流式/离散——流式=相邻轨道易，离散=跳间隔难）
    ('weighted_mf_score_per_sec', 10, 0.17),  # 加权多指协调分/秒（归一化跨度×指头数）
    ('discrete_mf_ratio', 0.3, 0.12),         # 离散型多指占比（norm_spread>1.5的比例）
    # 位置聚类
    ('position_cluster_count', 4, 0.12),   # 位置聚类数（虚拟轨道数）
    ('track_deviation_score', 0.3, 0.08),  # 离轨度（偏离轨道的平均距离）
    ('position_entropy', 2.0, 0.10),       # 位置熵（x轴分布均匀度，越高越不可预知）
    ('position_range_used', 0.5, 0.06),    # 位置范围占比（实际使用x范围/总宽）
    # 型切换
    ('pattern_switch_rate', 1.0, 0.10),    # 型切换频率（0.5s滑动窗口，加权烈度）
    ('direction_irregularity', 0.5, 0.08), # 方向不规则度（二阶方向变化熵）
    # 长条配置
    ('hold_interference_index', 0.3, 0.10),  # 长条干扰指数（降权重，更偏耐读而非配置）
    ('drag_flick_ratio', 0.3, 0.08),       # 滑/粉占比（Phigros特有操作类型）

    # === 耐力 (Stamina) ===
    ('stamina_ratio', 0.3, 0.15),          # 高负载窗口占比（≥平均×0.9）
    ('tap_per_second', 2.5, 0.12),         # 每秒tap数
    ('total_notes', 400, 0.06),            # 总物量
    ('tap_count', 400, 0.06),              # tap总数
    ('duration_sec', 100, 0.06),           # 时长（秒）
    ('rest_ratio', 0.3, 0.06),             # 休息段占比（>1s间隙累和/总时长）
    ('global_jack_count', 20, 0.06),       # 全局纵连计数（间隔<0.125拍）
    ('burst_intensity_mean', 0.3, 0.08),   # 爆发段平均密度
    ('tap_burst_top5', 0.5, 0.08),         # Top5 tap爆发密度

    # === 读谱 (Reading) ===
    ('density_transition_mean', 0.15, 0.08),  # 密度变化均值
    ('density_transition_std', 0.2, 0.08),    # 密度变化标准差
    ('tempo_change_count', 50, 0.08),         # 节奏突变次数
    ('offbeat_ratio', 0.04, 0.08),            # 不在拍点上的音符占比
    ('rhythm_entropy', 2.5, 0.06),            # 节奏熵
    ('type_switch_per_sec', 0.4, 0.06),       # 音符类型切换频率（Tap↔Flick↔Drag↔Hold）
    ('note_clutter_ratio', 0.05, 0.06),       # 视觉杂乱比率（紧密排列的不同位置音符）
]

# 动态cap参数：knee以下线性，超出部分power≤1轻微压缩，p95≈0特征的boost不放大
DC = {'knee': 1.2, 'power': 0.95}


def compute_simple_boost(feats, p95, p99):
    """特征平铺boost计算 — 不区分维度，全部放在一块算"""
    total_boost = _compute_dim_boost(feats, p95, p99, FLAT_FEATURES)
    total_boost = _dynamic_cap(total_boost)
    return total_boost, {'total_boost': round(total_boost, 4)}


def _dynamic_cap(raw):
    """动态cap：knee以下线性，超出部分^power放大(power>1.0)，帮助高难突破密度天花板"""
    KNEE = DC['knee']; POWER = DC['power']
    if raw <= KNEE:
        return raw
    excess = raw - KNEE
    return KNEE + excess ** POWER

# ====== 联合训练GB+boost ======
print('\n--- 联合训练 GB+boost ---')
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

# 删掉不再需要的 MUL_K
MUL_K = None

# ====== 全量评估 + boost分档统计 ======
print('\n' + '='*70)
print('  训练集内评估 + 分档统计')
print('='*70)

# 分档统计：对训练集按预测分数分档，记录每档boost的中位数和IQR
BINS = np.array([0,5,7,9,11,12,13,14,15,16,17,18,20])
n_bins = len(BINS) - 1
boost_per_bin = [[] for _ in range(n_bins)]

for i in range(n_samples):
    x = np.array([[feats_list[i].get(n,0) for n in gb_feature_names]])
    xs = scaler_gb.transform(x)
    p_gb = float(gb_full.predict(xs)[0])
    p_b, dims = compute_simple_boost(feats_list[i], p95_vals, p99_vals)
    p_f = p_gb + p_b
    print(f'{names_list[i]:<35} 真={labels[i]:.1f}  GB={p_gb:.3f}  +Boost={p_b:.3f}  ={p_f:.3f}  [{p_f-labels[i]:+.3f}]')
    
    # 分档
    for j in range(n_bins):
        if BINS[j] <= p_f < BINS[j+1]:
            boost_per_bin[j].append(float(p_b))
            break

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
    'dynamic_cap': {'knee': 1.2, 'power': 0.95},
    'boost_bin_stats': boost_bin_stats,
    'metrics': {'r2': r2, 'mae': mae, 'n_train': n_samples},
}
out_path = os.path.join(os.path.dirname(__file__), 'models', '6dim_model_v6_2.pkl')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'wb') as f:
    pickle.dump(model_out, f)
print(f'\n  模型已保存: {out_path}')

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
        p_b, dims = compute_simple_boost(feats_list[i], p95_vals, p99_vals)
        p_f = p_gb + p_b
        w.writerow([names_list[i], labels[i], round(p_gb,3), round(p_b,3), round(p_f,3), round(p_f-labels[i],3)])
print(f'  预测CSV已保存: {csv_path}')
print('\n' + '='*70)
print('  训练完成!')
print('='*70)