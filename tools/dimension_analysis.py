import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, numpy as np
from collections import defaultdict
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features, collect_all_notes, NOTE_HOLD, NOTE_TAP, NOTE_FLICK, NOTE_DRAG

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
sys.path.insert(0, os.path.dirname(__file__))
from predict_rpe import convert_rpe_to_standard

print('='*70)
print('  Phigros 难度5维度系统分析')
print('  参考: 交互/纵连 | 多押/多指 | 位移 | 稳定性/耐力 | 读谱')
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
            all_items.append({
                'folder': fn, 'filepath': info['levels'][lv],
                'difficulty': diffs[lv], 'level': lv,
            })

print(f'\n总谱面: {len(all_items)}')

# ====== 加载所有特征 ======
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
    if (i+1)%300==0: print(f'  已加载 {i+1}/{len(all_items)}')

feature_names = sorted(feats_list[0].keys())
X = np.array([[f.get(n,0) for n in feature_names] for f in feats_list])
y = np.array(labels)
lv_arr = np.array(levels_list)
n_feats = len(feature_names)
n_samples = len(feats_list)
print(f'\n成功加载: {n_samples} 张谱面, {n_feats} 个特征')
print(f'难度范围: {y.min():.1f} ~ {y.max():.1f}')

# ====== 计算所有特征与难度的相关系数 ======
corrs = []
for j, name in enumerate(feature_names):
    col = X[:, j]
    if np.std(col) > 0 and np.std(y) > 0:
        c = np.corrcoef(col, y)[0, 1]
    else:
        c = 0
    corrs.append((name, c, float(np.mean(col)), float(np.std(col))))

corrs.sort(key=lambda x: -abs(x[1]))

# ====== 五大维度定义 ======
# 基于社区研究+特征分析，将现有特征映射到5个维度

DIMENSIONS = {
    '交互_纵连': {
        'description': '两只手指交替点击(交互) + 同一轨道连续击打(纵连)',
        'features': [
            'jack_count', 'jack_ratio',
            'micro_max_0.0625beat', 'micro_max_0.125beat',
            'micro_peak_top5_0.0625beat', 'micro_peak_top5_0.125beat',
            'micro_spike_ratio_0.0625beat',
            'tap_burst_top5', 'tap_burst_05_top5', 'tap_burst_05_max',
            'tap_burst_peak_to_mean', 'hand_speed_index',
            'tap_per_second', 'tap_per_beat',
            'very_short_interval_ratio', 'short_interval_ratio',
            'core_micro_max_0.125beat', 'core_micro_top5_0.125beat',
            'extreme_tap_window_ratio',
            'peak_tap_density_4beat', 'mean_tap_density_4beat',
            'miniburst_count', 'miniburst_density',
        ],
        'direction': 1,
    },
    '多押_多指': {
        'description': '多根手指同时击打多个音符',
        'features': [
            'multi_finger_3plus_events', 'multi_finger_4plus_events',
            'multi_finger_3plus_ratio',
            'max_simultaneous', 'avg_simultaneous',
            'simultaneous_event_count', 'simultaneous_ratio',
            'chord_2note_ratio', 'chord_3note_ratio', 'chord_4plus_ratio',
            'sim_pos_spread_mean', 'sim_pos_spread_max',
            'mf_burst_count', 'mf_burst_avg_notes', 'mf_burst_max_notes',
            'mf_burst_avg_len_beats', 'mf_burst_max_len_beats',
            'multi_finger_density', 'multi_finger_max_simultaneous',
            'visual_complexity',
        ],
        'direction': 1,
    },
    '位移': {
        'description': '手部移动距离和模式',
        'features': [
            'avg_movement', 'max_movement', 'movement_per_second',
            'burst_avg_movement', 'burst_max_movement',
            'wide_jump_count', 'wide_jump_density',
            'position_std', 'position_range', 'position_iqr',
            'spread_balance',
            'note_clutter_count', 'note_clutter_ratio',
            'hold_lock_avg_displacement', 'hold_lock_max_displacement',
            'hold_lock_displacement_per_sec',
            'burst_movement_ratio',
        ],
        'direction': 1,
    },
    '稳定性_耐力': {
        'description': '持续输出能力和节奏稳定性',
        'features': [
            'sustained_density_run_count', 'sustained_density_run_ratio',
            'burst_window_count', 'max_consecutive_burst',
            'burst_intensity_mean',
            'high_density_ratio_1beat', 'high_density_duration_ratio_1beat',
            'high_density_ratio_4beat', 'high_density_duration_ratio_4beat',
            'high_density_ratio_16beat',
            'duration_sec', 'total_notes', 'notes_per_second',
            'stop_go_count', 'stop_go_ratio',
            'density_above_zero_ratio',
            'peak_density_16beat', 'mean_density_16beat',
            'std_density_1beat', 'std_density_2beat',
            'peak_density_1beat',
            'interval_cv',
        ],
        'direction': 1,
    },
    '读谱': {
        'description': '谱面可读性/视奏难度',
        'features': [
            'density_transition_mean', 'density_transition_max', 'density_transition_std',
            'speed_change_total_impact', 'speed_change_max_impact', 'speed_change_mean_impact',
            'speed_std', 'speed_range', 'speed_event_count',
            'tempo_change_count', 'tempo_change_ratio',
            'rhythm_entropy', 'rhythm_diversity', 'distinct_rhythm_count',
            'dominant_rhythm_ratio',
            'offbeat_ratio', 'weak_beat_ratio',
            'notes_above_ratio', 'notes_below_ratio',
            'position_entropy',
            'track_section_count', 'track_section_ratio',
            'cross_hand_event_count', 'cross_hand_ratio',
            'hold_tap_overlap_count', 'hold_tap_overlap_ratio',
            'hold_lock_tap_events', 'hold_lock_tap_events_per_hold',
            'max_concurrent_holds', 'avg_concurrent_holds',
            'hold_interference_index',
        ],
        'direction': 1,
    },
}

# ====== 计算每个维度的综合得分 ======
def compute_dimension_score(feats, dim_name, global_stats):
    dim = DIMENSIONS[dim_name]
    feat_list = dim['features']
    scores = []
    for fname in feat_list:
        if fname not in feats: continue
        if fname not in global_stats: continue
        mean_v, std_v = global_stats[fname]
        if std_v < 0.001: continue
        val = feats.get(fname, 0)
        z = (val - mean_v) / std_v
        corr = global_stats.get(f'{fname}_corr', 0)
        scores.append(z * abs(corr))
    if not scores:
        return 0
    raw = np.mean(scores) * 2.0
    if dim['direction'] == -1:
        raw = -raw
    return float(raw)

corrs_dict = {name: c for name, c, _, _ in corrs}
global_stats = {}
for j, name in enumerate(feature_names):
    col = X[:, j]
    global_stats[name] = (float(np.mean(col)), float(np.std(col)))
    c = corrs_dict.get(name, 0)
    global_stats[f'{name}_corr'] = c

# 计算每张谱面的5维得分
dim_scores_list = []
for i, feats in enumerate(feats_list):
    scores = {}
    for dim_name in DIMENSIONS:
        scores[dim_name] = compute_dimension_score(feats, dim_name, global_stats)
    dim_scores_list.append(scores)

# ====== 输出：每个特征维度与难度的相关性 ======
print('\n' + '='*70)
print('  各维度特征与难度相关系数 TOP特征')
print('='*70)
for dim_name, dim_info in DIMENSIONS.items():
    print(f'\n  【{dim_name}】- {dim_info["description"]}')
    dim_feats = [(n, c) for n, c, _, _ in corrs if n in dim_info['features']]
    dim_feats.sort(key=lambda x: -abs(x[1]))
    for name, c in dim_feats[:8]:
        arrow = '↑' if c > 0 else '↓'
        print(f'    {arrow} {name:35s} r={c:+.4f}')

# ====== 5维综合得分与难度相关系数 ======
print('\n' + '='*70)
print('  5维综合得分 vs 真实难度 相关系数')
print('='*70)
dim_y_corrs = []
for dim_name in DIMENSIONS:
    scores = np.array([s[dim_name] for s in dim_scores_list])
    if np.std(scores) > 0:
        c = float(np.corrcoef(scores, y)[0, 1])
    else:
        c = 0
    dim_y_corrs.append((dim_name, c))
    print(f'  {dim_name:15s}: r={c:+.4f}')

# ====== 多元线性回归R² ======
from sklearn.linear_model import LinearRegression
X_dim = np.array([[s[dn] for dn in DIMENSIONS] for s in dim_scores_list])
lr = LinearRegression()
lr.fit(X_dim, y)
y_pred_lr = lr.predict(X_dim)
from sklearn.metrics import r2_score
r2_dim = r2_score(y, y_pred_lr)
print(f'\n  5维线性回归 R² = {r2_dim:.4f}')
print(f'  各维度权重:')
for name, coef in zip(DIMENSIONS.keys(), lr.coef_):
    print(f'    {name:15s}: {coef:+.4f}')

# ====== 社区参考谱面详细分析 ======
print('\n' + '='*70)
print('  社区参考谱面详细分析')
print('='*70)

ref_charts = [
    ('Cthugha_IN', 16.0),
    ('Cthugha_AT', 16.1),
    ('Another Me_IN', 15.6),
    ('Inferno City_IN', 15.7),
]

# 先看一下名字格式
print('\n  参考谱面查找:')
for ref_name, ref_diff in ref_charts:
    prefix = ref_name.split('_')[0].lower()
    lv = ref_name.split('_')[1]
    candidates = []
    for i, (name, lvl) in enumerate(zip(names_list, levels_list)):
        if lvl == lv and prefix in name.lower():
            candidates.append((i, name, y[i]))
    print(f'  {ref_name}: 候选={[(n[:40], f"定数{d:.1f}") for _, n, d in candidates]}')
    idx = candidates[0][0] if candidates else None
    if idx is None:
        print(f'\n  {ref_name}: 未找到')
        continue
    feats = feats_list[idx]
    print(f'\n  {ref_name} (定数{ref_diff}):')
    print(f'    --- 5维得分 ---')
    for dim_name in DIMENSIONS:
        score = dim_scores_list[idx][dim_name]
        print(f'    {dim_name:15s}: {score:+.2f}')
    print(f'    --- 关键特征值(P95排名) ---')
    for name, c, _, _ in corrs[:15]:
        if abs(c) < 0.3: continue
        val = feats.get(name, 0)
        p95 = np.percentile(X[:, feature_names.index(name)], 95) if name in feature_names else 1
        rank_pct = np.sum(X[:, feature_names.index(name)] <= val) / n_samples * 100 if name in feature_names else 0
        print(f'    {name:35s} = {val:8.2f}  (P95={p95:8.2f}, 排名前{100-rank_pct:.0f}%)')

# ====== 按难度等级看各维度贡献 ======
print('\n' + '='*70)
print('  各难度等级 5维平均得分')
print('='*70)
for lv in ['EZ', 'HD', 'IN', 'AT']:
    mask = lv_arr == lv
    if np.sum(mask) < 3: continue
    print(f'\n  {lv} ({np.sum(mask)}张谱面):')
    for dim_name in DIMENSIONS:
        avg_s = np.mean([dim_scores_list[i][dim_name] for i in range(n_samples) if lv_arr[i] == lv])
        print(f'    {dim_name:15s}: {avg_s:+.2f}')

# ====== 高难谱特征画像 ======
print('\n' + '='*70)
print('  高难谱面(难度>=15.5) 5维画像')
print('='*70)
high_mask = y >= 15.5
low_mask = (y >= 10) & (y < 13)
dim_names_list = list(DIMENSIONS.keys())
high_scores_arr = np.array([[dim_scores_list[i][dn] for dn in dim_names_list] for i in range(n_samples) if high_mask[i]])
low_scores_arr = np.array([[dim_scores_list[i][dn] for dn in dim_names_list] for i in range(n_samples) if low_mask[i]])
high_scores_mean = np.mean(high_scores_arr, axis=0) if high_scores_arr.size > 0 else np.zeros(len(dim_names_list))
low_scores_mean = np.mean(low_scores_arr, axis=0) if low_scores_arr.size > 0 else np.zeros(len(dim_names_list))
print(f'\n  高难谱(≥15.5, n={np.sum(high_mask)})  vs  中难度(10~13, n={np.sum(low_mask)}):')
for i, dim_name in enumerate(dim_names_list):
    diff = high_scores_mean[i] - low_scores_mean[i]
    print(f'  {dim_name:15s}: 高难={high_scores_mean[i]:+.2f}  中={low_scores_mean[i]:+.2f}  差距={diff:+.2f}')

# ====== 偏差最大的谱面分析 ======
print('\n' + '='*70)
print('  各难度等级内 特征与难度相关性 top10')
print('='*70)
for lv in ['EZ', 'HD', 'IN', 'AT']:
    mask = lv_arr == lv
    if np.sum(mask) < 5: continue
    y_lv = y[mask]
    X_lv = X[mask]
    lv_corrs = []
    for j, name in enumerate(feature_names):
        col = X_lv[:, j]
        if np.std(col) > 0 and np.std(y_lv) > 0:
            c = np.corrcoef(col, y_lv)[0, 1]
        else:
            c = 0
        lv_corrs.append((name, c))
    lv_corrs.sort(key=lambda x: -abs(x[1]))
    print(f'\n  {lv} ({np.sum(mask)}张):')
    for name, c in lv_corrs[:10]:
        print(f'    r={c:+.4f}  {name}')

# ====== 总结 ======
print('\n' + '='*70)
print('  总结')
print('='*70)
print(f'\n  全量数据特征与难度相关系数 TOP 20:')
for name, c, mean_v, std_v in corrs[:20]:
    print(f'  r={c:+.4f}  {name}')
print(f'\n  5维线性回归 R² = {r2_dim:.4f}')
print(f'\n  提示: 下一步可以基于这5个维度重新设计特征，')
print(f'  加入社区分类标签(键盘谱/卡手谱/体力谱/读谱谱等)，')
print(f'  以及考虑"双指拆多押"等手法对难度的实际影响。')

# 保存分析结果
save_dir = os.path.join(os.path.dirname(__file__), 'analysis')
os.makedirs(save_dir, exist_ok=True)

analysis_data = {
    'dimension_defs': DIMENSIONS,
    'corrs_top50': [(name, c) for name, c, _, _ in corrs[:50]],
    'dim_correlations': dim_y_corrs,
    'dim_weights': list(zip(DIMENSIONS.keys(), lr.coef_)),
    'dim_r2': r2_dim,
}
with open(os.path.join(save_dir, 'dimension_analysis.json'), 'w') as f:
    json.dump(analysis_data, f, indent=2, ensure_ascii=False)

# 输出每张谱面的5维得分
with open(os.path.join(save_dir, 'all_chart_dimensions.csv'), 'w', encoding='utf-8') as f:
    dim_names = list(DIMENSIONS.keys())
    f.write('name,level,difficulty,' + ','.join(dim_names) + '\n')
    for i, name in enumerate(names_list):
        f.write(f'{name},{lv_arr[i]},{y[i]:.1f}')
        for dn in dim_names:
            f.write(f',{dim_scores_list[i][dn]:.4f}')
        f.write('\n')

print(f'\n  分析结果保存至: {save_dir}')
print('='*70)
