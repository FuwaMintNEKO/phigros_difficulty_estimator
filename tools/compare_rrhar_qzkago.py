import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)

rrhar = None
qzkago = None
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    if 'AT' not in info['levels'] or 'AT' not in diffs: continue
    lower = fn.lower()
    if 'rrhar' in lower:
        cd = load_chart_json(info['levels']['AT'])
        rrhar = {'name': fn, 'true': diffs['AT'], 'feats': extract_features(cd)}
    if 'qzkago' in lower:
        cd = load_chart_json(info['levels']['AT'])
        qzkago = {'name': fn, 'true': diffs['AT'], 'feats': extract_features(cd)}

if not rrhar or not qzkago:
    print('没找到')
    print('rrhar:', rrhar is not None)
    print('qzkago:', qzkago is not None)
    exit()

print('='*85)
print(f'{"特征":40s} {"Rrhar\'il(卡手)":>15s} {"QZKago(键盘)":>15s} {"Q/R比值":>8s}')
print('-'*85)

all_keys = sorted(rrhar['feats'].keys())
differences = []

for key in all_keys:
    rv = rrhar['feats'].get(key, 0)
    qv = qzkago['feats'].get(key, 0)
    if abs(rv) < 0.0001 and abs(qv) < 0.0001:
        continue
    ratio = qv / rv if abs(rv) > 0.0001 else (999 if qv > 0 else -999)
    differences.append((key, rv, qv, ratio))

# 按Q/R比值排序 - Q >> R 的特征（键盘谱比卡手多很多）
differences.sort(key=lambda x: -x[3])
print(f'\n--- QZKago >> Rrhar (键盘谱更多) ---')
for k, rv, qv, ratio in differences[:25]:
    if ratio > 2.0:
        print(f'  {k:40s} {rv:15.4f} {qv:15.4f} {ratio:8.1f}x')

# 按R/Q比值排序 - R >> Q 的特征（卡手谱比键盘多很多）
differences.sort(key=lambda x: x[3])
print(f'\n--- Rrhar >> QZKago (卡手谱更多)  ← 这就是"卡手"特征！ ---')
for k, rv, qv, ratio in differences[:25]:
    if abs(ratio) < 0.5 or (rv > 0.01 and qv < rv * 0.3):
        print(f'  {k:40s} {rv:15.4f} {qv:15.4f} {qv/max(rv,0.001):8.3f}')

# 也用绝对值对比
print(f'\n--- 两者都有的特征，绝对值对比 ---')
for k, rv, qv, ratio in sorted(differences, key=lambda x: -max(x[1], x[2])):
    if abs(rv) > 0.1 and abs(qv) > 0.1 and 0.5 < ratio < 2.0:
        print(f'  {k:40s} {rv:15.4f} {qv:15.4f}')
        if sum(1 for _ in [1]) > 15: break

# 找最能区分两者的特征：逻辑卡手 = 有规律但要求苛刻
# 思路：interval_cv(间隔变异系数大=乱), 密度突变, 锁手, 交叉手, 定轨precision
# 顺手：主导节奏占比高=重复pattern, 节奏熵低=规整
print(f'\n--- 关键区分特征 ---')
key_candidates = [
    'dominant_rhythm_ratio', 'rhythm_entropy', 'rhythm_diversity',
    'distinct_rhythm_count', 'interval_cv', 'density_transition_max',
    'density_transition_mean', 'stop_go_count', 'dense_mf_count',
    'cross_hand_event_count', 'cross_hand_ratio', 'hold_lock_tap_events',
    'hold_lock_avg_displacement', 'hold_lock_displacement_per_sec',
    'track_section_count', 'note_clutter_count', 'note_clutter_ratio',
    'tempo_change_ratio', 'tempo_change_count', 'short_interval_ratio',
    'position_entropy', 'position_std', 'burst_avg_movement', 'burst_max_movement',
    'offbeat_ratio', 'weak_beat_ratio', 'spread_balance',
    'multi_finger_3plus_events', 'multi_finger_4plus_events',
    'mf_events_per_second', 'mf_burst_count', 'mf_burst_avg_notes',
    'hold_tap_overlap_count', 'hold_tap_overlap_ratio',
    'hand_speed_index', 'notes_per_second', 'tap_per_second',
    'tap_burst_peak_to_mean', 'tap_burst_top5', 'sustained_density_run_count',
    'speed_event_count', 'speed_change_total_impact', 'speed_max',
    'wide_jump_count', 'wide_jump_density',
]
for k in key_candidates:
    rv = rrhar['feats'].get(k, 0)
    qv = qzkago['feats'].get(k, 0)
    ratio = qv / rv if abs(rv) > 1e-4 else 999
    print(f'  {k:40s} Rrhar={rv:12.4f}  QZKago={qv:12.4f}  Q/R={ratio:8.2f}')
