import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, numpy as np
from collections import defaultdict
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
sys.path.insert(0, os.path.dirname(__file__))

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

print(f'总谱面: {len(all_items)}')

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

feature_names = sorted(feats_list[0].keys())
X = np.array([[f.get(n,0) for n in feature_names] for f in feats_list])
y = np.array(labels)
lv_arr = np.array(levels_list)
n_samples = len(feats_list)

print(f'成功加载: {n_samples} 张谱面')

# 所有特征与难度的相关系数
corrs = []
for j, name in enumerate(feature_names):
    col = X[:, j]
    if np.std(col) > 0 and np.std(y) > 0:
        c = np.corrcoef(col, y)[0, 1]
    else:
        c = 0
    corrs.append((name, c))
corrs.sort(key=lambda x: -abs(x[1]))
corrs_dict = {name: c for name, c in corrs}

# ====== 五大维度特征映射（基于社区研究） ======
DIMENSIONS = {
    '交互_纵连': {
        'desc': '两只手指交替点击 + 同一轨道连续击打',
        'community': 'Class Memories(交互), BGA(纵连)是典型; 快速交互考验稳定性, 慢速交互考验协调性',
        'feats': [
            'jack_count', 'jack_ratio', 'micro_max_0.0625beat', 'micro_max_0.125beat',
            'micro_peak_top5_0.0625beat', 'micro_peak_top5_0.125beat', 'tap_burst_top5',
            'tap_burst_05_top5', 'tap_burst_05_max', 'tap_burst_peak_to_mean',
            'hand_speed_index', 'tap_per_second', 'tap_per_beat',
            'very_short_interval_ratio', 'short_interval_ratio',
            'core_micro_max_0.125beat', 'core_micro_top5_0.125beat',
            'extreme_tap_window_ratio', 'peak_tap_density_4beat', 'mean_tap_density_4beat',
            'miniburst_count', 'miniburst_density',
        ],
    },
    '多押_多指': {
        'desc': '多根手指同时击打多个音符; 三押是最考验协调性的键型',
        'community': 'Burn IN14(modulus)是经典三押谱; 多押辅助亮色标记可降低难度',
        'feats': [
            'multi_finger_3plus_events', 'multi_finger_4plus_events', 'multi_finger_3plus_ratio',
            'max_simultaneous', 'avg_simultaneous', 'simultaneous_event_count', 'simultaneous_ratio',
            'chord_2note_ratio', 'chord_3note_ratio', 'chord_4plus_ratio',
            'sim_pos_spread_mean', 'sim_pos_spread_max', 'mf_burst_count', 'mf_burst_avg_notes',
            'mf_burst_max_notes', 'mf_burst_avg_len_beats', 'mf_burst_max_len_beats',
            'multi_finger_density', 'multi_finger_max_simultaneous', 'visual_complexity',
        ],
    },
    '位移': {
        'desc': '打完上一个键手需要怎么移动; 含大跳、定位能力要求',
        'community': 'Spasmodic的单位移交互考验定位; 双位移交互(单乱)极难',
        'feats': [
            'avg_movement', 'max_movement', 'movement_per_second',
            'burst_avg_movement', 'burst_max_movement', 'wide_jump_count', 'wide_jump_density',
            'position_std', 'position_range', 'position_iqr', 'spread_balance',
            'note_clutter_count', 'note_clutter_ratio',
            'hold_lock_avg_displacement', 'hold_lock_max_displacement',
            'hold_lock_displacement_per_sec', 'burst_movement_ratio',
        ],
    },
    '稳定性_耐力': {
        'desc': '持续输出能力; 长曲高密度+少休息=体力谱',
        'community': 'Cthugha AT=体力谱(1444notes全程8分); Another Me IN=物量+少休息; Inferno City=体力谱(无休)',
        'feats': [
            'sustained_density_run_count', 'sustained_density_run_ratio',
            'burst_window_count', 'max_consecutive_burst', 'burst_intensity_mean',
            'high_density_ratio_1beat', 'high_density_duration_ratio_1beat',
            'high_density_ratio_4beat', 'high_density_duration_ratio_4beat',
            'high_density_ratio_16beat', 'duration_sec', 'total_notes', 'notes_per_second',
            'stop_go_count', 'stop_go_ratio', 'density_above_zero_ratio',
            'peak_density_16beat', 'mean_density_16beat',
            'std_density_1beat', 'std_density_2beat', 'peak_density_1beat', 'interval_cv',
        ],
    },
    '读谱': {
        'desc': '谱面可读性/视奏难度; 变速+密度突变+判定线变化+脑裂配置',
        'community': 'IEO AT=神秘协调(读谱难); 内三角比外三角难读谱; 多面下落=协调难',
        'feats': [
            'density_transition_mean', 'density_transition_max', 'density_transition_std',
            'speed_change_total_impact', 'speed_change_max_impact', 'speed_change_mean_impact',
            'speed_std', 'speed_range', 'speed_event_count',
            'tempo_change_count', 'tempo_change_ratio',
            'rhythm_entropy', 'rhythm_diversity', 'distinct_rhythm_count', 'dominant_rhythm_ratio',
            'offbeat_ratio', 'weak_beat_ratio', 'notes_above_ratio', 'notes_below_ratio',
            'position_entropy', 'track_section_count', 'track_section_ratio',
            'cross_hand_event_count', 'cross_hand_ratio',
            'hold_tap_overlap_count', 'hold_tap_overlap_ratio',
            'hold_lock_tap_events', 'hold_lock_tap_events_per_hold',
            'max_concurrent_holds', 'avg_concurrent_holds', 'hold_interference_index',
        ],
    },
}

# ====== 计算每个维度的Z-score综合得分 ======
global_stats = {}
for j, name in enumerate(feature_names):
    col = X[:, j]
    global_stats[name] = (float(np.mean(col)), float(np.std(col)))

def dim_score(feats, dim_name):
    feats_list_d = DIMENSIONS[dim_name]['feats']
    scores = []
    for fname in feats_list_d:
        if fname not in feats or fname not in global_stats: continue
        mean_v, std_v = global_stats[fname]
        if std_v < 0.001: continue
        z = (feats.get(fname, 0) - mean_v) / std_v
        c = abs(corrs_dict.get(fname, 0))
        scores.append(z * c)
    if not scores: return 0
    return float(np.mean(scores)) * 2.0

dim_scores = [{dn: dim_score(feats_list[i], dn) for dn in DIMENSIONS} for i in range(n_samples)]

# ====== 1. 找出每张谱面的优势维度 ======
print('\n' + '='*85)
print('  一、社区参考谱面详细5维画像')
print('='*85)

# 找到正确的文件夹名
def find_chart(partial_name, level):
    for i, (name, lvl) in enumerate(zip(names_list, levels_list)):
        if lvl == level and partial_name.lower() in name.lower():
            return i, name
    return None, None

ref_list = [
    ('Cthugha', 'IN', 'Cthugha USAO IN16.0 - 体力谱，早期16分+微纵连+1333notes'),
    ('Cthugha', 'AT', 'Cthugha USAO AT16.1 - 全程8分敲击+1444notes超体力谱'),
    ('AnotherMe', 'IN', 'Another Me IN15.6 - 片手8分+少休息物量谱(1449notes)'),
]

# 找Rrhar'il和QZKago
for i, (name, lvl) in enumerate(zip(names_list, levels_list)):
    if lvl == 'AT':
        if 'rrhar' in name.lower() or 'Rrhar' in name:
            ref_list.append(('Rrhar\'il', 'AT', f'Rrhar\'il AT17.6 - 卡手谱(锁手+位移+多指), {name}'))
        if 'qzkago' in name.lower():
            ref_list.append(('QZKago', 'AT', f'QZKago AT16.8 - 键盘谱(bpm214+多押+速度变化)'))

seen = set()
for partial, lv, comment in ref_list:
    idx, full_name = find_chart(partial, lv)
    if idx is None:
        print(f'\n  [{partial}_{lv}] 未找到')
        continue
    if idx in seen: continue
    seen.add(idx)
    
    feats = feats_list[idx]
    print(f'\n  [{partial}_{lv}] 定数={y[idx]:.1f}')
    print(f'  社区分类: {comment}')
    print(f'  文件夹: {full_name}')
    print(f'  5维得分:')
    for dn in DIMENSIONS:
        s = dim_scores[idx][dn]
        bar = '█' * max(0, int((s + 2) * 8)) + '░' * max(0, 16 - max(0, int((s+2)*8)))
        print(f'    {dn:12s} {s:+.2f} {bar}')
    
    print(f'  异常突出特征(>P95):')
    for name, c in corrs[:30]:
        if abs(c) < 0.3: continue
        val = feats.get(name, 0)
        p95_val = np.percentile(X[:, feature_names.index(name)], 95) if name in feature_names else 0
        if val > p95_val:
            rank = np.sum(X[:, feature_names.index(name)] <= val) / n_samples * 100
            print(f'    {name:35s} = {val:9.2f} (P95={p95_val:7.2f}, 超过{rank:.0f}%谱面)')

# ====== 2. 按社区分类范式 ======
print('\n' + '='*85)
print('  二、5维特征相关性汇总')
print('='*85)

print(f'\n  各维度内特征与难度平均相关系数:')
for dn in DIMENSIONS:
    dim_corrs = [abs(corrs_dict.get(f, 0)) for f in DIMENSIONS[dn]['feats'] if f in corrs_dict]
    print(f'  {dn:12s}: 平均|r|={np.mean(dim_corrs):.3f}  (共{len(DIMENSIONS[dn]["feats"])}个特征)')

# ====== 3. 难度级别之间维度差异 ======
print('\n' + '='*85)
print('  三、难度级别之间的维度变化')
print('='*85)

for lv in ['EZ', 'HD', 'IN', 'AT']:
    mask = lv_arr == lv
    if np.sum(mask) < 3: continue
    avg = {dn: np.mean([dim_scores[i][dn] for i in range(n_samples) if lv_arr[i] == lv]) for dn in DIMENSIONS}
    print(f'\n  {lv} ({np.sum(mask)}张):')
    for dn in DIMENSIONS:
        bar = '█' * max(0, int((avg[dn] + 2) * 6))
        print(f'    {dn:12s} {avg[dn]:+.2f} {bar}')

# ====== 4. AT内部难度区分 ======
print('\n' + '='*85)
print('  四、AT内部(45张) 特征与难度相关系数 TOP15')
print('='*85)
at_mask = lv_arr == 'AT'
X_at = X[at_mask]
y_at = y[at_mask]
at_corrs = []
for j, name in enumerate(feature_names):
    col = X_at[:, j]
    if np.std(col) > 0 and np.std(y_at) > 0:
        c = np.corrcoef(col, y_at)[0, 1]
    else:
        c = 0
    at_corrs.append((name, c))
at_corrs.sort(key=lambda x: -abs(x[1]))
print(f'\n  AT内部与难度正相关最强:')
for name, c in at_corrs[:15]:
    if c > 0:
        print(f'    r={c:+.4f}  {name}')
print(f'\n  AT内部与难度负相关最强(降难度因素):')
neg_count = 0
for name, c in at_corrs:
    if c < -0.2:
        print(f'    r={c:+.4f}  {name}')
        neg_count += 1
    if neg_count >= 5: break

# ====== 5. 特定特征分类 ======
print('\n' + '='*85)
print('  五、Phigros社区分类对应的量化指标')
print('='*85)
print('''
  基于社区研究，谱面可分为以下类型：
  
  【键盘谱】= 高BPM + 密集交互 + 多押但布局顺手
    代表: QZKago Requiem, Stardust:RAY
    特征: 高 hand_speed_index, 高 multi_finger_3plus_events,
          低 hold_lock_tap_events, 低 burst_avg_movement
  
  【卡手谱】= 锁手 + 别扭位移 + hold干扰
    代表: Rrhar'il, 彩
    特征: 高 hold_lock_tap_events, 高 hold_lock_displacement,
          高 burst_avg_movement, 高 hold_tap_overlap_ratio
  
  【体力谱】= 长曲 + 高密度 + 少休息
    代表: Cthugha AT, Another Me IN, Inferno City IN
    特征: 高 sustained_density_run_count, 高 total_notes,
          低 stop_go_ratio, 高 duration_sec
  
  【读谱谱】= 变速 + 密度突变 + 多面下落 + 脑裂
    代表: IEO AT, 望影の方舟Six, Sigma
    特征: 高 density_transition_max, 高 speed_change_total_impact,
          高 offbeat_ratio, 高 rhythm_entropy
  
  【纵连谱】= 同轨快速连打
    代表: BGA, Cross Soul
    特征: 高 jack_count, 高 miniburst_count, 高 micro_max
''')

# ====== 6. 特征筛选 ======
print('='*85)
print('  六、关键特征推荐 (非冗余, 高区分度)')
print('='*85)

# 按维度选出5个最有代表性的特征
selected = {}
for dn in DIMENSIONS:
    feats_list_d = DIMENSIONS[dn]['feats']
    ranked = sorted([(abs(corrs_dict.get(f,0)), f) for f in feats_list_d], reverse=True)
    selected[dn] = ranked[:5]
    print(f'\n  {dn}:')
    for c, fname in ranked[:5]:
        print(f'    |r|={c:.4f}  {fname}')

# ====== 7. 全量预测 5维模型 vs 当前模型 ======
print('\n' + '='*85)
print('  七、当前外推公式 vs 5维模型的对比建议')
print('='*85)

# 计算每个Chart的5维总分
X_dim = np.array([[dim_scores[i][dn] for dn in DIMENSIONS] for i in range(n_samples)])
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
lr = LinearRegression()
lr.fit(X_dim, y)
y_pred_dim = lr.predict(X_dim)
r2_dim = r2_score(y, y_pred_dim)

print(f'''
  当前5维线性模型 R² = {r2_dim:.4f}
  
  各维度实际权重（从全量数据学到的）:
''')
for name, coef in zip(DIMENSIONS.keys(), lr.coef_):
    print(f'    {name:12s}: {coef:+.4f}')

print(f'''
  基于社区研究+全量数据分析的建议:
  
  1. 交互/纵连维度 权重最大 — 手速是最基础的难度指标
     建议保留: jack_count, tap_burst_top5, micro_max_0.0625beat
  
  2. 稳定性/耐力维度 权重第二大 — 高难谱的核心区分因素
     建议保留: sustained_density_run_count, notes_per_second,
              high_density_duration_ratio_16beat
  
  3. 位移维度 — 卡手谱与键盘谱的区分关键
     建议保留: burst_avg_movement, wide_jump_count, hold_lock_displacement
  
  4. 读谱维度 — AT内部最重要的区分因素之一
     建议保留: density_transition_max, tempo_change_count,
              speed_change_total_impact, rhythm_entropy
  
  5. 多押/多指维度 — 对双指玩家难度加成大
     但注意: "双指拆多押"手法会降低实际难度
     建议保留: multi_finger_3plus_events (但需要打折系数)
  
  关键改进建议:
  - 引入"键盘/卡手"分类系数: 对卡手谱的位移/lock特征加权重
  - 区分"双指可拆"的多押 vs "强制多指"的多押
  - 耐力维度应加入"休息密度"(连续高密度段的长度)
  - 读谱维度需要区分"变速读谱" vs "配置读谱"
''')

# ====== 8. 保存结果 ======
save_dir = os.path.join(os.path.dirname(__file__), 'analysis')
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, 'dimension_analysis.json'), 'w') as f:
    json.dump({
        'dimension_corrs': {dn: np.mean([abs(corrs_dict.get(f,0)) for f in DIMENSIONS[dn]['feats'] if f in corrs_dict]) for dn in DIMENSIONS},
        'r2_5dim_linear': r2_dim,
        'dim_weights': {dn: float(coef) for dn, coef in zip(DIMENSIONS.keys(), lr.coef_)},
    }, f, indent=2, ensure_ascii=False)

# 输出关键特征推荐
print(f'  分析完成，结果保存在 {save_dir}')
print('='*85)
