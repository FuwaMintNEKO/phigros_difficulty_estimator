# -*- coding: utf-8 -*-
"""v11 实验1: 官谱歌曲分组CV 各段 bias + 上架谱 16+ 明细"""
import os, sys, json, pickle, numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from boost_config import MANUAL_FLAT

# ===== 与 train_final_v10.py 相同的配置 =====
CAPS = {'_default': 4.0}
FLAT = MANUAL_FLAT
LOWW = 1.5
CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)
all_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
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
           'core_peak_density_1sec_top5avg', 'core_peak_density_top5avg_1beat',
           'movement_per_second', 'movement_density_index'}
gb_feature_names = [nn for nn in feature_names
                    if nn in GB_KEEP or not any(kw in nn for kw in GB_EXCLUDE_KEYWORDS)]
print(f'GB特征数: {len(gb_feature_names)}')

X_base = np.array([[f.get(nn, 0) for nn in gb_feature_names] for f in feats_list])
y = np.array(labels)
LV_ORDER = ['EZ', 'HD', 'IN_AT']
X_lv = np.zeros((n, len(LV_ORDER)))
for i, lv in enumerate(levels_list):
    key = 'IN_AT' if lv in ('IN', 'AT') else lv
    X_lv[i, LV_ORDER.index(key)] = 1.0
levels_arr = np.array(levels_list)
SAMPLE_W = np.ones(n)
SAMPLE_W[levels_arr == 'EZ'] = LOWW
SAMPLE_W[levels_arr == 'HD'] = LOWW

def compute_boost_v9(feats, p95_vals, p99_vals):
    total = 0.0
    cap = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0)
        pv = p95_vals.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t: continue
        e = v / t - 1.0
        c = CAPS.get(fname, cap)
        if c is not None and e > c: e = c
        x = co * (e ** 0.70)
        if v > max(p99_vals.get(fname, 0), bl * 0.5):
            pe = v / max(p99_vals.get(fname, 0), bl * 0.5) - 1.0
            if c is not None and pe > c: pe = c
            x += co * max(0, pe) ** 0.70 * 0.5
        total += x
    return total

# ===== 歌曲分组5折CV + bias =====
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_base, y, groups=np.array(names_list)))
oof = np.zeros(n)
for fi, (tr, te) in enumerate(splits):
    p95_vals, p99_vals = {}, {}
    for j, name in enumerate(gb_feature_names):
        col = X_base[tr, j]
        p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
        p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
    boosts = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in feats_list])
    X_tr = np.hstack([X_base[tr], X_lv[tr]])
    X_te = np.hstack([X_base[te], X_lv[te]])
    sc = StandardScaler().fit(X_tr)
    gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
    gb.fit(sc.transform(X_tr), y[tr] - boosts[tr], sample_weight=SAMPLE_W[tr])
    oof[te] = gb.predict(sc.transform(X_te)) + boosts[te]
    print(f'  fold{fi} OOF完成', flush=True)

errs = oof - y
print(f'\n===== 官谱歌曲分组CV =====')
print(f'整体MAE = {mean_absolute_error(y, oof):.4f} | 整体bias = {errs.mean():+.4f}')
for lv in ['EZ', 'HD', 'IN', 'AT']:
    mk = np.where(levels_arr == lv)[0]
    print(f'  {lv}: n={len(mk)} MAE={mean_absolute_error(y[mk], oof[mk]):.4f} bias={errs[mk].mean():+.4f}')
# 定数段细分
for lo, hi, tag in [(11, 13, '11-13'), (13, 14, '13-14'), (14, 15, '14-15'), (15, 16, '15-16'), (16, 16.5, '16-16.5'), (16.5, 99, '16.5+')]:
    mk = np.where((y >= lo) & (y < hi))[0]
    if len(mk):
        print(f'  定数[{tag}]: n={len(mk)} MAE={mean_absolute_error(y[mk], oof[mk]):.4f} bias={errs[mk].mean():+.4f}')
# 16+ 按多指分组 (官谱)
print('\n=== 官谱 16+ 段: 多指/双指分组 bias (真值=官谱) ===')
for lo in [16.0, 16.5]:
    mk = np.where(y >= lo)[0]
    if len(mk) == 0: continue
    g_mf = [i for i in mk if feats_list[i].get('multi_finger_3plus_events', 0) >= 30]
    g_df = [i for i in mk if feats_list[i].get('multi_finger_3plus_events', 0) <= 5]
    g_rest = [i for i in mk if i not in g_mf and i not in g_df]
    for g, tag in [(g_mf, '多指(mf3>=30)'), (g_df, '双指(mf3<=5)'), (g_rest, '混合')]:
        if len(g):
            print(f'  >= {lo} [{tag}]: n={len(g)} bias={errs[g].mean():+.4f} MAE={mean_absolute_error(y[g], oof[g]):.4f}')
print('DONE')
