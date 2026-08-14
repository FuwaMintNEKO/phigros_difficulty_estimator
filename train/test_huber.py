# -*- coding: utf-8 -*-
"""v10c: 测试 Huber/Quantile 损失是否对 gimmick 离群点更稳健 (仅V10a设计+损失变体)"""
import os, sys, numpy as np, time
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from boost_config import MANUAL_FLAT

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
feats_list, labels, levels_list, names_list = [], [], [], []
for item in all_items:
    try:
        cd = load_chart_json(item['filepath'])
        fe = extract_features(cd)
        if fe:
            feats_list.append(fe); labels.append(item['difficulty'])
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
    'speed_change_total', 'micro_max_', 'micro_spike_',
    'density_above_zero', 'core_density_above_zero',
    'density_skew', 'density_transition_max',
    'avg_hold_duration', 'max_hold_duration', 'finger_vs_total',
]
GB_KEEP = {'density_dimension', 'real_core_notes_per_second',
           'core_peak_density_1sec_top5avg', 'core_peak_density_top5avg_1beat'}
gb_feature_names = [nn for nn in feature_names
                    if nn in GB_KEEP or not any(kw in nn for kw in GB_EXCLUDE_KEYWORDS)]
X_base = np.array([[f.get(nn, 0) for nn in gb_feature_names] for f in feats_list])
y = np.array(labels)
LV_ORDER = ['EZ','HD','IN','AT']
X_lv = np.zeros((n, 4))
for i, lv in enumerate(levels_list):
    X_lv[i, LV_ORDER.index(lv)] = 1.0

def compute_boost_v9(feats, p95_vals, p99_vals):
    total = 0.0
    for fname, bl, co in MANUAL_FLAT:
        v = feats.get(fname, 0)
        pv = p95_vals.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t: continue
        e = v / t - 1.0
        x = co * (e ** 0.70)
        if v > max(p99_vals.get(fname, 0), bl * 0.5):
            pe = v / max(p99_vals.get(fname, 0), bl * 0.5) - 1.0
            x += co * max(0, pe) ** 0.70 * 0.5
        total += x
    return total

kf = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kf.split(X_base))

def run(loss, label, extra_kw=None):
    extra_kw = extra_kw or {}
    oof = np.zeros(n)
    t0 = time.time()
    for fi, (tr, te) in enumerate(splits):
        p95_vals, p99_vals = {}, {}
        for j, name in enumerate(gb_feature_names):
            col = X_base[tr, j]
            p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
            p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
        boosts = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in feats_list])
        X_tr = np.hstack([X_base[tr], X_lv[tr]]); X_te = np.hstack([X_base[te], X_lv[te]])
        sc = StandardScaler().fit(X_tr)
        gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                       learning_rate=0.05, subsample=0.8, random_state=42,
                                       loss=loss, **extra_kw)
        gb.fit(sc.transform(X_tr), y[tr] - boosts[tr])
        oof[te] = gb.predict(sc.transform(X_te)) + boosts[te]
    err = oof - y
    print(f'{label:<24} MAE={mean_absolute_error(y, oof):.4f} 偏差={np.mean(err):+.3f} '
          f'R2={r2_score(y, oof):.4f} RMSE={np.sqrt(np.mean(err**2)):.4f} 耗时={(time.time()-t0)/60:.1f}min')
    return oof

print('='*70)
oof_sq = run('squared_error', 'squared (v10a对照)')
oof_h = run('huber', 'huber alpha=1.0')
oof_h3 = run('huber', 'huber alpha=3.0')
# 分档对比 huber vs squared
print('\n=== 按定数档位 MAE: squared vs huber(3.0) ===')
for name, lo, hi in [('<4',0,4),('4-7',4,7),('7-11',7,11),('11-14',11,14),('14-16.5',14,16.5),('>16.5',16.5,99)]:
    m = np.where((y >= lo) & (y < hi))[0]
    if len(m)==0: continue
    print(f'  [{name}]: n={len(m):<3} squared={mean_absolute_error(y[m], oof_sq[m]):.3f} '
          f'huber3={mean_absolute_error(y[m], oof_h3[m]):.3f}')

np.savez(os.path.join(_ROOT, 'tools', 'cv_oof_v10c.npz'),
         oof_sq=oof_sq, oof_huber=oof_h, oof_huber3=oof_h3, y=y,
         names=np.array(names_list), levels=np.array(levels_list))
