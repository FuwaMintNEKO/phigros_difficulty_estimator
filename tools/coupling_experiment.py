# -*- coding: utf-8 -*-
"""耦合特征实验: 音符-线事件耦合 vs V10 基线
诚实 5 折 CV (与 train_final_v10 相同流程: GB残差+boost+level)

新增耦合特征 (从 chart_data 独立计算):
  c_note_move_win    每个音符 ±96ticks 窗内同线移动事件数均值
  c_note_rotate_win  每个音符 ±96ticks 窗内同线旋转事件数均值
  c_note_speed_win   每个音符 ±96ticks 窗内同线变速事件数均值
  c_main_line_ratio  主线(音符最多线)音符占比
  c_main_move_dens   主线移动事件密度 (事件/秒)
  c_note_line_count  有音符的判定线数量
  c_move_on_note_ratio 移动事件发生在有音符的线上的比例
"""
import os, sys, numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from boost_config import MANUAL_FLAT

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)
all_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties:
        continue
    diffs = song_difficulties[sid]
    for lv in ['EZ', 'HD', 'IN', 'AT']:
        if lv in info['levels'] and lv in diffs:
            all_items.append({'folder': fn, 'filepath': info['levels'][lv],
                              'difficulty': diffs[lv], 'level': lv})
print(f'官谱总数: {len(all_items)}')

def coupling_features(cd):
    """从 chart_data 计算耦合特征 dict (numpy 二分查找, 快)"""
    W = 96  # ticks 窗口 (半拍)
    jls = cd.get('judgeLineList', [])
    line_notes = [[] for _ in jls]
    line_note_cnt = [0] * len(jls)
    fallback_bpm = jls[0].get('bpm', 120.0) if jls else 120.0
    for li, line in enumerate(jls):
        for n in line.get('notes', []) + line.get('notesAbove', []) + line.get('notesBelow', []):
            line_notes[li].append(n.get('time', 0))
            line_note_cnt[li] += 1
    if not any(line_notes):
        return None
    total_notes = sum(line_note_cnt)
    note_move = note_rot = note_speed = 0.0
    for li in range(len(jls)):
        notes = np.asarray(sorted(line_notes[li]), dtype=np.float64)
        if len(notes) == 0:
            continue
        for ev_key in ('judgeLineMoveEvents', 'judgeLineRotateEvents', 'speedEvents'):
            evs = np.asarray([e.get('startTime', 0) for e in jls[li].get(ev_key, [])],
                             dtype=np.float64)
            if len(evs) == 0:
                continue
            # 每个音符窗口 [t-W, t+W] 内事件数 → 二分
            left = np.searchsorted(evs, notes - W, side='left')
            right = np.searchsorted(evs, notes + W, side='right')
            cnt = right - left
            if ev_key == 'judgeLineMoveEvents':
                note_move += cnt.sum()
            elif ev_key == 'judgeLineRotateEvents':
                note_rot += cnt.sum()
            else:
                note_speed += cnt.sum()
    main_li = int(np.argmax(line_note_cnt))
    main_ratio = line_note_cnt[main_li] / total_notes
    all_times = np.concatenate([np.asarray(v, dtype=np.float64) for v in line_notes if len(v)])
    ds = max((np.percentile(all_times, 99) / (fallback_bpm / 1.875)) if fallback_bpm else 1, 1.0)
    main_move_dens = len(jls[main_li].get('judgeLineMoveEvents', [])) / ds
    note_lines = sum(1 for c in line_note_cnt if c > 0)
    all_move = sum(len(jl.get('judgeLineMoveEvents', [])) for jl in jls)
    move_on_note = sum(len(jls[li].get('judgeLineMoveEvents', []))
                       for li in range(len(jls)) if line_note_cnt[li] > 0)
    return {
        'c_note_move_win': note_move / total_notes,
        'c_note_rotate_win': note_rot / total_notes,
        'c_note_speed_win': note_speed / total_notes,
        'c_main_line_ratio': main_ratio,
        'c_main_move_dens': main_move_dens,
        'c_note_line_count': note_lines,
        'c_move_on_note_ratio': move_on_note / max(all_move, 1),
    }

feats_list, labels, levels_list, names_list = [], [], [], []
coup_list = []
for item in all_items:
    try:
        cd = load_chart_json(item['filepath'])
        feats = extract_features(cd)
        cf = coupling_features(cd)
        if feats and cf:
            feats_list.append(feats); labels.append(item['difficulty'])
            levels_list.append(item['level']); names_list.append(item['folder'])
            coup_list.append(cf)
    except Exception:
        pass
n = len(feats_list)
print(f'特征提取成功: {n}')

feature_names = sorted(feats_list[0].keys())
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
GB_KEEP = {'density_dimension', 'real_core_notes_per_second',
           'core_peak_density_1sec_top5avg', 'core_peak_density_top5avg_1beat'}
gb_feature_names = [nn for nn in feature_names
                    if nn in GB_KEEP or not any(kw in nn for kw in GB_EXCLUDE_KEYWORDS)]

y = np.array(labels)
LV_ORDER = ['EZ', 'HD', 'IN', 'AT']
X_lv = np.zeros((n, 4))
for i, lv in enumerate(levels_list):
    X_lv[i, LV_ORDER.index(lv)] = 1.0
levels_arr = np.array(levels_list)
COUP_NAMES = sorted(coup_list[0].keys())
X_coup = np.array([[cf[nn] for nn in COUP_NAMES] for cf in coup_list])
print(f'GB特征: {len(gb_feature_names)}, 耦合特征: {len(COUP_NAMES)}')
print('耦合特征示例(前5谱):')
for cf in coup_list[:5]:
    print('  ', {k: round(v, 3) for k, v in cf.items()})

def compute_boost_v9(feats, p95_vals, p99_vals):
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

def run_cv(X_extra, name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(n)
    for tr, te in kf.split(X_base):
        p95_vals, p99_vals = {}, {}
        for j, nm in enumerate(gb_feature_names):
            col = X_base[tr, j]
            p95_vals[nm] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
            p99_vals[nm] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
        boosts = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in feats_list])
        X_tr = np.hstack([X_base[tr], X_lv[tr], X_extra[tr]])
        X_te = np.hstack([X_base[te], X_lv[te], X_extra[te]])
        sc = StandardScaler().fit(X_tr)
        gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                       learning_rate=0.05, subsample=0.8, random_state=42)
        gb.fit(sc.transform(X_tr), y[tr] - boosts[tr])
        oof[te] = gb.predict(sc.transform(X_te)) + boosts[te]
    mae = mean_absolute_error(y, oof)
    r2 = r2_score(y, oof)
    print(f'  {name}: OOF MAE={mae:.4f} R2={r2:.4f}')
    return mae, oof

X_base = np.array([[f.get(nn, 0) for nn in gb_feature_names] for f in feats_list])
print('\n===== 5折CV对比 =====')
# 1. 基线 V10 (无耦合)
mae0, oof0 = run_cv(np.zeros((n, 0)), 'V10 基线(无耦合)')
# 2. V10 + 耦合特征
mae1, oof1 = run_cv(X_coup, 'V10+耦合特征')
print(f'\n改善: {mae0:.4f} -> {mae1:.4f} ({(mae1 - mae0) / mae0 * 100:+.2f}%)')
# 3. 按level分解
print('\n按level MAE (基线 -> +耦合):')
for lv in LV_ORDER:
    m = np.where(levels_arr == lv)[0]
    if len(m):
        print(f'  {lv}: n={len(m)} {mean_absolute_error(y[m], oof0[m]):.4f} -> {mean_absolute_error(y[m], oof1[m]):.4f}')
