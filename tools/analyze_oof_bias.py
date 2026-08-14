# -*- coding: utf-8 -*-
"""OOF(袋外)偏差分析: 用5折CV的验证折预测, 无泄漏地看官方谱偏差模式
+ 自制谱(有定数)偏差分析
输出: 整体/按level/按定数桶 的有符号偏差; 低估/高估top; 特征对比
"""
import os, sys, pickle, re
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from boost_config import MANUAL_FLAT
from unified_parser import load_chart_from_bytes
import app

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
LV_ORDER = ['EZ', 'HD', 'IN', 'AT']

# ===== 加载官方谱 (与训练脚本一致) =====
charts = find_chart_files(CHART_DIR)
diffs = load_difficulty_tsv(TSV)
all_items = []
for fn, info in charts.items():
    for lv in LV_ORDER:
        if lv not in info['levels']:
            continue
        d = (diffs.get(info['song_id']) or {}).get(lv)
        if d is None:
            continue
        all_items.append({'folder': fn, 'filepath': info['levels'][lv],
                          'difficulty': d, 'level': lv})
feats_list, labels, levels_list, names_list = [], [], [], []
for item in all_items:
    try:
        feats = extract_features(load_chart_json(item['filepath']))
        if feats:
            feats_list.append(feats); labels.append(item['difficulty'])
            levels_list.append(item['level']); names_list.append(item['folder'])
    except Exception:
        pass
n = len(feats_list)
print(f'官方谱样本: {n}')

# ===== boost 配置 (与正式模型一致: 当前 MANUAL_FLAT + cap4) =====
FLAT = MANUAL_FLAT
CAPS = {'_default': 4.0}

def compute_boost(feats, p95_vals, p99_vals):
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

X_base = np.array([[f.get(nn, 0) for nn in gb_feature_names] for f in feats_list])
y = np.array(labels)
X_lv = np.zeros((n, 4))
for i, lv in enumerate(levels_list):
    X_lv[i, LV_ORDER.index(lv)] = 1.0
levels_arr = np.array(levels_list)

# ===== 5折CV OOF =====
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(n)
for fi, (tr, te) in enumerate(kf.split(X_base)):
    p95_vals = {name: (float(np.percentile(X_base[tr, j], 95)) if np.max(X_base[tr, j]) > 0 else 0)
                for j, name in enumerate(gb_feature_names)}
    p99_vals = {name: (float(np.percentile(X_base[tr, j], 99)) if np.max(X_base[tr, j]) > 0 else 0)
                for j, name in enumerate(gb_feature_names)}
    boosts = np.array([compute_boost(f, p95_vals, p99_vals) for f in feats_list])
    X_tr = np.hstack([X_base[tr], X_lv[tr]])
    X_te = np.hstack([X_base[te], X_lv[te]])
    sc = StandardScaler().fit(X_tr)
    gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
    gb.fit(sc.transform(X_tr), y[tr] - boosts[tr])
    oof[te] = gb.predict(sc.transform(X_te)) + boosts[te]
    print(f'  fold{fi} OOF done', flush=True)

errs = oof - y
print(f'\n===== OOF 偏差分析 (无泄漏, 当前模型配置) =====')
print(f'整体 MAE={mean_absolute_error(y, oof):.4f}  有符号={np.mean(errs):+.4f}')

print('\n===== 按 level =====')
for lv in LV_ORDER:
    m = np.where(levels_arr == lv)[0]
    if len(m):
        print(f'{lv:>3} n={len(m):>4} MAE={mean_absolute_error(y[m], oof[m]):.4f} 有符号={np.mean(errs[m]):+.4f}')

print('\n===== 按真实定数桶 =====')
bins = [(0, 8), (8, 10), (10, 12), (12, 13.5), (13.5, 15), (15, 16), (16, 99)]
for lo, hi in bins:
    m = np.where((y >= lo) & (y < hi))[0]
    if len(m):
        print(f'[{lo:>4},{hi:>4}) n={len(m):>4} MAE={mean_absolute_error(y[m], oof[m]):.4f} 有符号={np.mean(errs[m]):+.4f}')

print('\n===== 低估最严重 top12 (预测过低) =====')
idx = np.argsort(errs)[:12]
for i in idx:
    print(f'{names_list[i][:38]:<38} {levels_list[i]:>3} 真={y[i]:>5.1f} 预测={oof[i]:>5.1f} 差={errs[i]:+.2f}')
print('\n===== 高估最严重 top12 (预测过高) =====')
idx = np.argsort(errs)[-12:][::-1]
for i in idx:
    print(f'{names_list[i][:38]:<38} {levels_list[i]:>3} 真={y[i]:>5.1f} 预测={oof[i]:>5.1f} 差={errs[i]:+.2f}')

print('\n===== 特征对比: 低估组(<-.6) vs 高估组(>+.6) =====')
under_i = np.where(errs < -0.6)[0]
over_i = np.where(errs > 0.6)[0]
print(f'低估组 n={len(under_i)}, 高估组 n={len(over_i)}')
KEY_F = ['real_core_notes_per_second', 'above_avg_density_mean', 'weighted_mf_score_per_sec',
         'chord_alternation_rate', 'type_switch_per_sec', 'above_avg_duration_sec',
         'tempo_change_count', 'speed_volatility', 'jline_movement_density',
         'jline_rotate_density', 'hold_interference_index', 'density_transition_std',
         'note_clutter_ratio', 'pattern_switch_rate', 'rhythm_entropy',
         'direction_irregularity', 'stair_complexity', 'stair_speed_avg',
         'note_speed_non1_ratio', 'chord_jack_density', 'avg_chord_size_poly',
         'above_below_cross', 'discrete_mf_ratio', 'position_range_used']
print(f'{"特征":<28} {"低估均值":>10} {"高估均值":>10} {"低估-高估":>10}')
for kf in KEY_F:
    u = np.mean([f.get(kf, 0) for f in np.array(feats_list)[under_i]])
    o = np.mean([f.get(kf, 0) for f in np.array(feats_list)[over_i]])
    print(f'{kf:<28} {u:>10.3f} {o:>10.3f} {u-o:>+10.3f}')

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_oof_rows.pkl'), 'wb') as f:
    pickle.dump({'names': names_list, 'levels': levels_list, 'y': y, 'oof': oof, 'errs': errs,
                 'feats': feats_list}, f)
print('\n已保存 _oof_rows.pkl')
