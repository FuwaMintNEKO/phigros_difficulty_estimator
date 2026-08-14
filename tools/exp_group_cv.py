# -*- coding: utf-8 -*-
"""泛化能力实验: 随机5折CV (有同曲泄漏, 与train_final_v10一致) vs 按歌曲分组5折CV (真实新谱泛化)
结论意义: 对自制谱(新曲)预测, 分组CV才是诚实误差; 若分组CV远差于随机CV, 说明模型靠"同曲其他难度"泄漏。
"""
import os, sys, pickle
import numpy as np

ROOT = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator'
sys.path.insert(0, ROOT)

from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from boost_config import MANUAL_FLAT

# ===== 数据加载 (与 train_final_v10 一致) =====
CHART_DIR = os.path.join(ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(ROOT, 'data', 'info', 'difficulty.tsv')
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
    'note_speed', 'flash_', 'visible_time', 'chord_jack', 'fast_hold',
]
GB_KEEP = {'density_dimension', 'real_core_notes_per_second',
           'core_peak_density_1sec_top5avg', 'core_peak_density_top5avg_1beat'}
gb_feature_names = [nn for nn in feature_names
                    if nn in GB_KEEP or not any(kw in nn for kw in GB_EXCLUDE_KEYWORDS)]
print(f'GB特征数: {len(gb_feature_names)}')

X_base = np.array([[f.get(nn, 0) for nn in gb_feature_names] for f in feats_list])
y = np.array(labels)
LV_ORDER = ['EZ', 'HD', 'IN', 'AT']
X_lv = np.zeros((n, 4))
for i, lv in enumerate(levels_list):
    X_lv[i, LV_ORDER.index(lv)] = 1.0
levels_arr = np.array(levels_list)
groups = np.array([fn for fn in names_list])  # 歌曲名作为组

# ===== boost (训练fold内算p95/p99, 与 train_final_v10 一致; caps 对齐生产 --caps 4.0) =====
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

def run_cv(splits, tag):
    oof = np.zeros(n)
    for fi, (tr, te) in enumerate(splits):
        p95_vals, p99_vals = {}, {}
        for j, name in enumerate(gb_feature_names):
            col = X_base[tr, j]
            p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
            p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
        boosts = np.array([compute_boost(f, p95_vals, p99_vals) for f in feats_list])
        X_tr = np.hstack([X_base[tr], X_lv[tr]])
        X_te = np.hstack([X_base[te], X_lv[te]])
        sc = StandardScaler().fit(X_tr)
        gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                       learning_rate=0.05, subsample=0.8, random_state=42)
        gb.fit(sc.transform(X_tr), y[tr] - boosts[tr])
        oof[te] = gb.predict(sc.transform(X_te)) + boosts[te]
    mae = mean_absolute_error(y, oof)
    r2 = r2_score(y, oof)
    print(f'===== {tag} =====')
    print(f'整体MAE = {mae:.4f}  R2 = {r2:.4f}')
    for lv in LV_ORDER:
        m = np.where(levels_arr == lv)[0]
        if len(m):
            print(f'  {lv}: n={len(m)} MAE={mean_absolute_error(y[m], oof[m]):.4f}')
    # 按定数段
    print('  按定数段 (MAE / Bias):')
    for lo, hi in [(1,7),(7,11),(11,13),(13,14.5),(14.5,16),(16,17),(17,20)]:
        m = np.where((y >= lo) & (y < hi))[0]
        if len(m):
            e = oof[m] - y[m]
            print(f'    [{lo:4.1f},{hi:4.1f}): n={len(m):4d} MAE={mean_absolute_error(y[m], oof[m]):.4f} '
                  f'Bias={np.mean(e):+.4f} 中位={np.median(e):+.4f}')
    # 分组CV下最差的10个
    if tag.startswith('歌曲分组'):
        errs = oof - y
        worst = sorted(range(n), key=lambda i: -abs(errs[i]))[:10]
        print('  最差10个:')
        for i in worst:
            print(f'    {names_list[i][:40]:40s} [{levels_list[i]}] 真={y[i]:5.2f} 预={oof[i]:5.2f} d={errs[i]:+.2f}')
    return oof

# ===== 随机 5 折 CV =====
kf = KFold(n_splits=5, shuffle=True, random_state=42)
run_cv(list(kf.split(X_base)), '随机5折CV (含同曲泄漏)')

# ===== 按歌曲分组 5 折 CV =====
gkf = GroupKFold(n_splits=5)
run_cv(list(gkf.split(X_base, y, groups)), '歌曲分组5折CV (真实新谱泛化)')
