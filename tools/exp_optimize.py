# -*- coding: utf-8 -*-
"""优化实验: 在歌曲分组CV框架下对比
  基线 (V14逻辑+caps4.0) / Huber损失 / 低段样本加权 / 预测后分段校准
  特征先缓存到 data/phira/_feats_cache.npz 加速多次实验。
"""
import os, sys, pickle
import numpy as np

ROOT = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator'
sys.path.insert(0, ROOT)

from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features, collect_all_notes, time_to_seconds, _compute_duration_sec
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from boost_config import MANUAL_FLAT
import numpy as np

def compute_tail_features(chart_data):
    """尾杀特征: 末段(最后15%时长)的密度集中度与1秒峰值
    社区证据: DF AT 最后5秒"全游最难尾杀"; QZKago 尾杀20秒"拉高一大档"。
    """
    all_notes, judge_lines, bpm_timeline = collect_all_notes(chart_data)
    if not all_notes:
        return {}
    fallback_bpm = judge_lines[0].get('bpm', 120.0) if judge_lines else 120.0
    times = np.array([n['time'] for n in all_notes])
    types = np.array([n['type'] for n in all_notes])
    tsec = np.array([time_to_seconds(t, max(n.get('bpm', fallback_bpm), 1.0), bpm_timeline)
                     for t, n in zip(times, all_notes)])
    total_sec = _compute_duration_sec(bpm_timeline, times[-1] / 32.0)
    if total_sec <= 0:
        return {}
    core = (types == 1) | (types == 3)  # Tap+Hold
    cut = total_sec * 0.85
    tail_mask = tsec >= cut
    tail_core = tsec[tail_mask & core]
    out = {'tail_note_count': int(tail_mask.sum()),
           'tail_ratio': float(tail_mask.sum() / len(tsec))}
    if tail_core.size > 3 and (total_sec - cut) > 0.5:
        # 末段1秒滑动窗口峰值 (简化为窗口直方图)
        win = 1.0
        nb = max(int(np.ceil((total_sec - cut) / win)), 1)
        counts = np.zeros(nb)
        for t in tail_core:
            idx = int((t - cut) / win)
            if 0 <= idx < nb:
                counts[idx] += 1
        tail_peak_1s = float(counts.max())
        # 全局1秒峰值
        nb_all = max(int(np.ceil(total_sec / win)), 1)
        all_counts = np.zeros(nb_all)
        for t in tsec[core]:
            idx = int(t / win)
            if 0 <= idx < nb_all:
                all_counts[idx] += 1
        global_peak_1s = float(all_counts.max()) if all_counts.max() > 0 else 1.0
        global_mean_1s = float(np.mean(all_counts)) if all_counts.size else 0.0
        out['tail_peak_1s_ratio'] = tail_peak_1s / max(global_peak_1s, 1.0)  # 末段峰值/全局峰值
        out['tail_peak_vs_mean'] = tail_peak_1s / max(global_mean_1s, 0.01)  # 末段峰值/全局均值
        out['tail_density'] = float(tail_core.size / max(total_sec - cut, 0.01))  # 末段核心密度
        out['tail_core_share'] = float(tail_core.size / max(core.sum(), 1))  # 末段核心音符占比
    return out

CACHE = os.path.join(ROOT, 'data', 'phira', '_feats_cache.npz')

CHART_DIR = os.path.join(ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(ROOT, 'data', 'info', 'difficulty.tsv')

# ===== 特征缓存 =====
if os.path.exists(CACHE):
    print('加载特征缓存...')
    d = np.load(CACHE, allow_pickle=True)
    feats_list, labels, levels_list, names_list = (d['feats_list'], d['labels'],
                                                   d['levels_list'], d['names_list'])
    gb_feature_names = list(d['gb_feature_names'])
    n = len(feats_list)
else:
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
    feats_list, labels, levels_list, names_list = [], [], [], []
    for item in all_items:
        try:
            cd = load_chart_json(item['filepath'])
            feats = extract_features(cd)
            if feats:
                feats_list.append(feats); labels.append(item['difficulty'])
                levels_list.append(item['level']); names_list.append(item['folder'])
        except Exception:
            pass
    n = len(feats_list)
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
        'note_speed', 'flash_', 'visible_time', 'chord_jack', 'fast_hold',
    ]
    GB_KEEP = {'density_dimension', 'real_core_notes_per_second',
               'core_peak_density_1sec_top5avg', 'core_peak_density_top5avg_1beat'}
    gb_feature_names = [nn for nn in feature_names
                        if nn in GB_KEEP or not any(kw in nn for kw in GB_EXCLUDE_KEYWORDS)]
    np.savez(CACHE, feats_list=np.array(feats_list, dtype=object),
             labels=np.array(labels), levels_list=np.array(levels_list, dtype=object),
             names_list=np.array(names_list, dtype=object),
             gb_feature_names=np.array(gb_feature_names, dtype=object))
    print(f'特征缓存已保存: {CACHE}')

print(f'样本: {n}, GB特征: {len(gb_feature_names)}')

# ===== 尾杀特征缓存 (与主缓存对齐 by (folder, level)) =====
TAIL_CACHE = os.path.join(ROOT, 'data', 'phira', '_tail_cache.npz')
TAIL_NAMES = ['tail_note_count', 'tail_ratio', 'tail_peak_1s_ratio',
              'tail_peak_vs_mean', 'tail_density', 'tail_core_share']
if os.path.exists(TAIL_CACHE):
    print('加载尾杀特征缓存...')
    td = np.load(TAIL_CACHE, allow_pickle=True)
    tail_cols = {k: td[k] for k in TAIL_NAMES}
else:
    song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
    chart_files = find_chart_files(CHART_DIR)
    tail_map = {}
    for fn, info in chart_files.items():
        sid = info['song_id']
        if sid not in song_difficulties:
            continue
        diffs = song_difficulties[sid]
        for lv in ['EZ','HD','IN','AT']:
            if lv in info['levels'] and lv in diffs:
                try:
                    cd = load_chart_json(info['levels'][lv])
                    tail_map[(fn, lv)] = compute_tail_features(cd)
                except Exception:
                    tail_map[(fn, lv)] = {}
    tail_cols = {k: np.zeros(n) for k in TAIL_NAMES}
    for i in range(n):
        tf = tail_map.get((names_list[i], levels_list[i]), {})
        for k in TAIL_NAMES:
            tail_cols[k][i] = tf.get(k, 0)
    np.savez(TAIL_CACHE, names_list=np.array(names_list, dtype=object),
             levels_list=np.array(levels_list, dtype=object), **tail_cols)
    print(f'尾杀特征缓存已保存: {TAIL_CACHE}')
X_tail = np.column_stack([tail_cols[k] for k in TAIL_NAMES])
print(f'尾杀特征非零样本: {int((X_tail.sum(axis=1) > 0).sum())}/{n}')

y = np.array(labels)
LV_ORDER = ['EZ', 'HD', 'IN', 'AT']
orig_names = list(gb_feature_names)
gb_feature_names = orig_names + TAIL_NAMES
X_base = np.hstack([np.array([[f.get(nn, 0) for nn in orig_names] for f in feats_list]),
                    X_tail])
X_lv = np.zeros((n, 4))
for i, lv in enumerate(levels_list):
    X_lv[i, LV_ORDER.index(lv)] = 1.0
levels_arr = np.array(levels_list)
groups = np.array([fn for fn in names_list])
CAPS = {'_default': 4.0}

def compute_boost(feats, p95_vals, p99_vals):
    total = 0.0
    cap = CAPS.get('_default', None)
    for fname, bl, co in MANUAL_FLAT:
        v = feats.get(fname, 0)
        pv = p95_vals.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t:
            continue
        e = v / t - 1.0
        c = CAPS.get(fname, cap)
        if c is not None and e > c:
            e = c
        x = co * (e ** 0.70)
        if v > max(p99_vals.get(fname, 0), bl * 0.5):
            pe = v / max(p99_vals.get(fname, 0), bl * 0.5) - 1.0
            if c is not None and pe > c:
                pe = c
            x += co * max(0, pe) ** 0.70 * 0.5
        total += x
    return total

def run_cv(splits, loss='squared_error', sample_w=None, tag='', x_base=None, names=None):
    xb = X_base if x_base is None else x_base
    gn = gb_feature_names if names is None else names
    oof = np.zeros(n)
    for fi, (tr, te) in enumerate(splits):
        p95_vals, p99_vals = {}, {}
        for j, name in enumerate(gn):
            col = xb[tr, j]
            p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
            p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
        boosts = np.array([compute_boost(f, p95_vals, p99_vals) for f in feats_list])
        X_tr = np.hstack([xb[tr], X_lv[tr]])
        X_te = np.hstack([xb[te], X_lv[te]])
        sc = StandardScaler().fit(X_tr)
        gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                       learning_rate=0.05, subsample=0.8, random_state=42,
                                       loss=loss)
        w = sample_w[tr] if sample_w is not None else None
        gb.fit(sc.transform(X_tr), y[tr] - boosts[tr], sample_weight=w)
        oof[te] = gb.predict(sc.transform(X_te)) + boosts[te]
    print(f'{tag}整体MAE = {mean_absolute_error(y, oof):.4f}', end='')
    for lv in LV_ORDER:
        m = np.where(levels_arr == lv)[0]
        print(f'  {lv}={mean_absolute_error(y[m], oof[m]):.4f}', end='')
    print()
    return oof

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_base, y, groups))
print(f'歌曲分组5折CV, 组数={len(set(groups))}')

X_no_tail = X_base[:, :len(orig_names)]
# 基线 (无尾杀特征)
oof0 = run_cv(splits, tag='基线(无尾杀): ', x_base=X_no_tail, names=orig_names)
# 尾杀特征
oof_t = run_cv(splits, tag='基线(+尾杀): ')
# Huber (含尾杀)
oof1 = run_cv(splits, loss='huber', tag='Huber(+尾杀): ')
# 低段加权 (含尾杀)
w2 = np.ones(n)
w2[levels_arr == 'EZ'] = 1.5
w2[levels_arr == 'HD'] = 1.5
oof2 = run_cv(splits, sample_w=w2, tag='低段加权1.5(+尾杀): ')

# 预测后分段校准 (在 OOF 上拟合分段线性, 探索用)
def piecewise_cal(oof_in, y_in, breaks, tag=''):
    oof_c = np.zeros_like(oof_in)
    bounds = [-np.inf] + breaks + [np.inf]
    for i in range(len(bounds) - 1):
        m = (oof_in >= bounds[i]) & (oof_in < bounds[i + 1])
        if m.sum() < 5:
            oof_c[m] = oof_in[m]
            continue
        a, b = np.polyfit(oof_in[m], y_in[m], 1)
        oof_c[m] = a * oof_in[m] + b
    print(f'{tag}校准后整体MAE = {mean_absolute_error(y, oof_c):.4f}')
    for lv in LV_ORDER:
        mm = np.where(levels_arr == lv)[0]
        print(f'  {lv}={mean_absolute_error(y[mm], oof_c[mm]):.4f}', end='')
    print()
    return oof_c

print()
piecewise_cal(oof0, y, [7, 11, 14.5, 16.5], '分段校准[基线]: ')
