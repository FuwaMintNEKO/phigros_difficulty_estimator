"""全面特征分析：官谱特征有效性"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os, numpy as np
sys.path.insert(0, '.')
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFF_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

# 1. 加载官谱
song_diffs = load_difficulty_tsv(DIFF_TSV)
chart_files = find_chart_files(CHART_DIR)

all_feats = []
all_labels = []
all_names = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_diffs: continue
    diffs = song_diffs[sid]
    for lv in ['EZ','HD','IN','AT']:
        if lv in info['levels'] and lv in diffs:
            fp = info['levels'][lv]
            try:
                cd = load_chart_json(fp)
                fe = extract_features(cd)
                if fe:
                    all_feats.append(fe)
                    all_labels.append(diffs[lv])
                    all_names.append(f'{fn}.{lv}')
            except:
                pass

print(f'官谱加载: {len(all_feats)} 条')
y = np.array(all_labels)

# 2. 特征名列表 & 分组
feature_names = sorted(all_feats[0].keys())

# 按维度手动分组
DIM_GROUPS = {
    '1-配置/纵连': ['tap_micro_max_0.0625beat','tap_micro_top5_0.0625beat','tap_micro_max_0.125beat','tap_micro_top5_0.125beat',
                   'tap_micro_max_0.25beat','tap_micro_top5_0.25beat',
                   'tap_burst_top5','tap_burst_05_top5','tap_burst_05_max','tap_burst_peak_to_mean',
                   'short_jack_count','long_jack_count','jack_max_run','global_jack_count',
                   'same_line_jack_count','same_line_jack_ratio','miniburst_count','miniburst_density',
                   'very_short_interval_ratio','short_interval_ratio',
                   'finger_peak_tps','finger_avg_peak_tps','finger_vs_total_ratio',
                   'tap_per_second','tap_per_beat','max_consecutive_burst','burst_window_count','burst_window_ratio',
                   'extreme_tap_window_ratio','hand_speed_index'],
    '2-多指': ['multi_finger_3plus_events','multi_finger_4plus_events','multi_finger_3plus_ratio',
              'multi_finger_max_simultaneous','multi_finger_density','mf_burst_count','mf_burst_avg_notes',
              'mf_burst_max_notes','mf_burst_avg_len_beats','mf_burst_max_len_beats',
              'mf_events_per_second','mf_with_hold_count','mf_with_hold_ratio',
              'dense_mf_count','dense_mf_ratio','cross_hand_event_count','cross_hand_ratio',
              'cross_line_3plus_count','multi_line_sim_events','multi_line_sim_ratio',
              'avg_chord_size','chord_2note_ratio','chord_3note_ratio','chord_4plus_ratio',
              'max_simultaneous','avg_simultaneous','simultaneous_event_count','simultaneous_ratio',
              'sim_pos_spread_mean','sim_pos_spread_max','visual_complexity',
              'stair_event_count','stair_climb_count','stair_total_steps','stair_density'],
    '3-位移': ['wide_jump_count','wide_jump_density','burst_avg_movement','burst_max_movement','burst_movement_ratio',
              'hold_lock_displacement_per_sec','hold_lock_tap_events','hold_lock_tap_events_per_hold',
              'hold_lock_avg_displacement','hold_lock_max_displacement',
              'movement_per_second','total_movement','avg_movement','max_movement',
              'position_mean','position_std','position_range','position_abs_mean','position_iqr',
              'position_entropy','left_ratio','right_ratio','center_ratio','spread_balance'],
    '4-耐力': ['total_notes','tap_count','drag_count','hold_count','flick_count',
              'notes_per_second','notes_per_beat','tap_notes_per_second','tap_notes_per_beat',
              'duration_sec','duration_beats',
              'high_density_duration_ratio_16beat','high_density_duration_ratio_8beat',
              'high_density_duration_ratio_4beat','high_density_duration_ratio_2beat','high_density_duration_ratio_1beat',
              'high_density_ratio_16beat','high_density_ratio_8beat','high_density_ratio_4beat',
              'sustained_density_run_count','sustained_density_run_ratio',
              'total_hold_duration_beats','total_hold_duration_sec','hold_duration_ratio',
              'max_concurrent_holds','avg_concurrent_holds','concurrent_hold_events',
              'avg_hold_duration_beats','max_hold_duration_beats',
              'hold_tap_overlap_count','hold_tap_overlap_ratio',
              'note_clutter_count','note_clutter_ratio','density_above_zero_ratio',
              'max_gap_sec','stop_go_count','stop_go_ratio',
              'burst_intensity_mean'],
    '5-读谱': ['density_transition_mean','density_transition_max','density_transition_std','density_skew',
              'tempo_change_count','tempo_change_ratio','offbeat_ratio','weak_beat_ratio',
              'rhythm_entropy','rhythm_diversity','distinct_rhythm_count','dominant_rhythm_ratio',
              'speed_change_total_impact','speed_change_max_impact','speed_change_mean_impact',
              'speed_event_count','speed_mean','speed_std','speed_max','speed_min','speed_range',
              'bpm','bpm_change_count','bpm_min','bpm_max','bpm_range','bpm_std',
              'type_switch_ratio','type_switch_per_sec','tap_ratio','drag_ratio','hold_ratio','flick_ratio',
              'avg_interval_beats','std_interval_beats','min_interval_beats','interval_cv',
              'has_AT','judge_line_count','first_note_time','last_note_time',
              'notes_above_ratio','notes_below_ratio',
              'track_section_count','track_section_ratio',
              'peak_density_0.25beat','peak_density_0.5beat','peak_density_1beat','peak_density_2beat','peak_density_4beat','peak_density_8beat','peak_density_16beat',
              'peak_tap_density_4beat','mean_tap_density_4beat',
              'peak_density_top5avg_0.25beat','peak_density_top5avg_0.5beat','peak_density_top5avg_1beat'],
}

# 收集未分类特征
categorized = set()
for dim_name, flist in DIM_GROUPS.items():
    for f in flist:
        categorized.add(f)
uncategorized = [f for f in feature_names if f not in categorized]

print(f'\n总特征数: {len(feature_names)}')
print(f'已分类: {len(categorized)}')
print(f'未分类: {len(uncategorized)}')
for f in uncategorized:
    print(f'  [未分类] {f}')

# 3. 计算每个特征与难度的相关系数
print('\n' + '='*100)
print('维度分析: 各特征与官谱难度的 Pearson 相关系数')
print('='*100)

X = np.array([[f.get(n, 0) for n in feature_names] for f in all_feats])

all_corrs = []
for j, name in enumerate(feature_names):
    col = X[:, j]
    if np.std(col) > 0 and np.std(y) > 0:
        corr = float(np.corrcoef(col, y)[0, 1])
    else:
        corr = 0
    all_corrs.append((abs(corr), corr, name))

# 按维度输出
for dim_name, flist in DIM_GROUPS.items():
    dim_corrs = [(f, abs(np.corrcoef(
        X[:, feature_names.index(f)], y)[0, 1]) if np.std(X[:, feature_names.index(f)]) > 0 else 0)
                 for f in flist if f in feature_names]
    dim_corrs.sort(key=lambda x: -x[1])
    
    print(f'\n【{dim_name}】特征数={len(dim_corrs)}')
    print(f'  {"特征名":<35s} {"相关系数":>8s} {"在v4公式中":<12s} {"在v3公式中":<12s}')
    print('  ' + '-'*35 + ' ' + '-'*8 + ' ' + '-'*12 + ' ' + '-'*12)
    
    # v4使用的特征
    v4_used = {'tap_micro_max_0.0625beat','tap_micro_top5_0.0625beat','tap_burst_top5','short_jack_count','long_jack_count','jack_max_run',
               'tap_per_second','very_short_interval_ratio','tap_burst_05_top5','finger_peak_tps','finger_avg_peak_tps',
               'multi_finger_3plus_events','sim_pos_spread_max','cross_line_3plus_count','multi_line_sim_ratio','stair_total_steps','avg_chord_size',
               'wide_jump_count','burst_avg_movement','hold_lock_displacement_per_sec','movement_per_second','stair_event_count',
               'total_notes','tap_notes_per_second','notes_per_second','high_density_duration_ratio_16beat','sustained_density_run_count',
               'density_transition_max','tempo_change_count','speed_change_total_impact','offbeat_ratio','rhythm_entropy',
               'bpm_change_count','density_transition_mean','type_switch_ratio','type_switch_per_sec'}
    v3_used = {'tap_micro_max_0.0625beat','tap_micro_top5_0.0625beat','tap_burst_top5','jack_count','tap_per_second','very_short_interval_ratio','tap_burst_05_top5',
               'multi_finger_3plus_events','sim_pos_spread_max',
               'wide_jump_count','burst_avg_movement','hold_lock_displacement_per_sec','movement_per_second','hold_lock_tap_events_per_hold',
               'total_notes','tap_notes_per_second','notes_per_second','high_density_duration_ratio_16beat','sustained_density_run_count',
               'density_transition_max','tempo_change_count','speed_change_total_impact','offbeat_ratio','rhythm_entropy',
               'bpm_change_count','density_transition_mean','type_switch_ratio','type_switch_per_sec'}
    
    for fname, corr in dim_corrs[:15]:
        in_v4 = '✔' if fname in v4_used else ''
        in_v3 = '✔' if fname in v3_used else ''
        print(f'  {fname:<35s} {corr:>8.4f} {in_v4:<12s} {in_v3:<12s}')
    
    # 给出该维度推荐使用的特征（|r|>0.4且未使用的）
    strong_not_used = [(f, c) for f, c in dim_corrs if abs(c) > 0.4 and f not in v4_used]
    if strong_not_used:
        print(f'  >> 高相关但未使用的:')
        for f, c in strong_not_used:
            print(f'     {f:<35s} r={c:.4f}')

# 4. Top 20 最强相关特征
print('\n' + '='*100)
print('Top 30 最强相关特征（全量）')
print('='*100)
all_corrs.sort(key=lambda x: -x[0])
for rank, (abs_r, r, name) in enumerate(all_corrs[:30], 1):
    used = 'v4' if name in v4_used else ('v3' if name in v3_used else '')
    print(f'  {rank:2d}. {name:<35s} r={r:+.4f}  [{used}]')
