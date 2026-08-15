# -*- coding: utf-8 -*-
"""v12.8实验: 二次蒸馏 — 官谱982 + 3280首unranked(v12.7完整管线伪标签) 混合训练
伪标签老师=当前生产v12.7(含规则+校准+官谱偏移后的完整输出)
用法: python train/train_v12_distill.py
"""
import os, sys, io, csv, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from boost_config import MANUAL_FLAT
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from unified_parser import load_chart_from_bytes
import importlib
import app as app_mod
importlib.reload(app_mod)

FLAT = list(MANUAL_FLAT)
CAPS = {'_default': 4.0}
_JLINE_P95_FIX = {'jline_movement_density': 107.1, 'jline_rotate_density': 18.6, 'jline_disappear_density': 15.1}

def compute_boost(feats, p95_vals, p99_vals):
    total = 0.0
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0)
        pv = p95_vals.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t: continue
        e = min(v / t - 1.0, CAPS.get('_default', 4.0))
        x = co * (e ** 0.70)
        if v > max(p99_vals.get(fname, 0), bl * 0.5):
            pe = min(v / max(p99_vals.get(fname, 0), bl * 0.5) - 1.0, CAPS.get('_default', 4.0))
            x += co * max(0, pe) ** 0.70 * 0.5
        total += x
    return total

# ===== 官谱 =====
song_difficulties = load_difficulty_tsv(os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv'))
chart_files = find_chart_files(os.path.join(_ROOT, 'data', 'chart'))
official_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    for lv in ['EZ', 'HD', 'IN', 'AT']:
        if lv in info['levels'] and lv in diffs:
            official_items.append({'name': 'off|' + fn, 'lv': lv, 'diff': diffs[lv], 'path': info['levels'][lv]})
print(f'官谱: {len(official_items)}')

# ===== unranked 3280 =====
rows = list(csv.DictReader(open(os.path.join(_ROOT, 'data', 'phira', 'unranked_4star_list.csv'), encoding='utf-8-sig')))
sel = []
for r in rows:
    try:
        rt = float(r.get('rating', 0) or 0); rc = float(r.get('ratingCount', 0) or 0)
        df = float(r.get('difficulty', 0) or 0)
    except Exception:
        continue
    if rt >= 0.9 and rc >= 30 and 11 <= df <= 19.5 and abs(df - round(df)) >= 1e-6:
        sel.append(r)
print(f'unranked高质量: {len(sel)}')
JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star')

feats_list, labels, levels_list, names_list, weights = [], [], [], [], []
for it in official_items:
    try:
        cd = load_chart_json(it['path'])
        feats = extract_features(cd)
        if feats:
            feats_list.append(feats); labels.append(it['diff'])
            levels_list.append(it['lv']); names_list.append(it['name']); weights.append(1.0)
    except Exception:
        pass
n_off = len(feats_list)
n_unr = 0
for row in sel:
    p = os.path.join(JSON_DIR, row['id'] + '.json')
    if not os.path.exists(p):
        continue
    try:
        with open(p, 'rb') as f:
            cd, raw = load_chart_from_bytes(f.read())
        feats = extract_features(cd, speed=1.0)
        if not feats:
            continue
        lv = 'AT' if 'AT' in (row['level'] or '').upper() else ('IN' if 'IN' in (row['level'] or '').upper() else 'HD')
        res = app_mod.predict_one_chart(cd, speed=1.0, level=lv, chart_name=row['id'])
        if isinstance(res, tuple): res = res[0]
        pred = res.get('prediction') if res else None
        if pred is None or pred < 9 or pred > 20:
            continue
        feats_list.append(feats); labels.append(float(pred))
        levels_list.append(lv); names_list.append('unr|' + str(row['id'])); weights.append(0.5)
        n_unr += 1
    except Exception:
        pass
print(f'特征: 官谱{n_off} + unranked伪标签{n_unr}')
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
           'core_peak_density_1sec_top5avg', 'core_peak_density_top5avg_1beat',
           'movement_per_second', 'movement_density_index',
           'jline_move_disp_per_sec', 'jline_rotate_disp_per_sec', 'jline_hidden_time_ratio',
           'hold_lock_weighted_per_sec', 'hold_lock_weighted_per_hold'}
gb_feature_names = [nn for nn in feature_names
                    if nn in GB_KEEP or not any(kw in nn for kw in GB_EXCLUDE_KEYWORDS)]
print(f'GB特征数: {len(gb_feature_names)}')
X_base = np.array([[f.get(nn, 0) for nn in gb_feature_names] for f in feats_list])
y = np.array(labels)
W = np.array(weights)
LV_ORDER = ['EZ', 'HD', 'IN_AT']
X_lv = np.zeros((n, len(LV_ORDER)))
for i, lv in enumerate(levels_list):
    key = 'IN_AT' if lv in ('IN', 'AT') else lv
    X_lv[i, LV_ORDER.index(key)] = 1.0

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_base, y, groups=np.array(names_list)))
oof = np.zeros(n)
for fi, (tr, te) in enumerate(splits):
    p95, p99 = {}, {}
    for j, nm in enumerate(gb_feature_names):
        col = X_base[tr, j]
        p95[nm] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
        p99[nm] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
    for k, v in _JLINE_P95_FIX.items():
        if k in p95: p95[k] = v
    boosts = np.array([compute_boost(f, p95, p99) for f in feats_list])
    X_tr = np.hstack([X_base[tr], X_lv[tr]])
    X_te = np.hstack([X_base[te], X_lv[te]])
    sc = StandardScaler().fit(X_tr)
    gb = GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_leaf=3,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
    gb.fit(sc.transform(X_tr), y[tr] - boosts[tr], sample_weight=W[tr])
    oof[te] = gb.predict(sc.transform(X_te)) + boosts[te]
    print(f'  fold{fi} 完成', flush=True)
off_mask = np.array([nm.startswith('off|') for nm in names_list])
print(f'官谱子集CV: MAE={mean_absolute_error(y[off_mask], oof[off_mask]):.4f} bias={(oof[off_mask]-y[off_mask]).mean():+.4f}')
print(f'全体CV: MAE={mean_absolute_error(y, oof):.4f} bias={(oof-y).mean():+.4f}')

p95, p99 = {}, {}
for j, nm in enumerate(gb_feature_names):
    col = X_base[:, j]
    p95[nm] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
    p99[nm] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
for k, v in _JLINE_P95_FIX.items():
    if k in p95: p95[k] = v
boosts = np.array([compute_boost(f, p95, p99) for f in feats_list])
X_all = np.hstack([X_base, X_lv])
scaler = StandardScaler().fit(X_all)
gb_final = GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_leaf=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
gb_final.fit(scaler.transform(X_all), y - boosts, sample_weight=W)
model = {'gb': gb_final, 'scaler': scaler, 'feature_names': gb_feature_names,
         'p95_vals': p95, 'p99_vals': p99, 'lv_order': LV_ORDER,
         'version': 'v12.8-distill', 'n_train': n, 'MANUAL_FLAT': FLAT}
path = os.path.join(_ROOT, 'models', '6dim_model_v12_distill.pkl')
with open(path, 'wb') as f:
    pickle.dump(model, f)
print(f'已保存: {path}')
