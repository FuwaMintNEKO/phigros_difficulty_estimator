# -*- coding: utf-8 -*-
"""v12.6社区尺度诊断: 纯高质量unranked谱(社区定数标签)训练GB残差模型 → 预测官谱对照
目的: 找出社区标尺与官谱标尺的特征定价差异 (社区模型预测官谱的偏差模式 = 两个尺度的分歧)
用法: python train/train_v12_community.py
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

# ===== unranked 训练集 =====
train_list = []
with open(os.path.join(_ROOT, 'data', 'phira', 'train_unranked_1000.csv'), encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        train_list.append(row)
print(f'unranked训练集: {len(train_list)} 首')
JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star')
feats_list, labels, levels_list, names_list = [], [], [], []
for row in train_list:
    p = os.path.join(JSON_DIR, row['id'] + '.json')
    if not os.path.exists(p):
        continue
    try:
        with open(p, 'rb') as f:
            cd, raw = load_chart_from_bytes(f.read())
        feats = extract_features(cd, speed=1.0)
        if feats:
            lv = 'AT' if 'AT' in (row['level'] or '').upper() else ('IN' if 'IN' in (row['level'] or '').upper() else 'HD')
            feats_list.append(feats); labels.append(float(row['diff']))
            levels_list.append(lv); names_list.append(str(row['id']))
    except Exception:
        pass
n = len(feats_list)
print(f'特征成功: {n}')
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
LV_ORDER = ['EZ', 'HD', 'IN_AT']
X_lv = np.zeros((n, len(LV_ORDER)))
for i, lv in enumerate(levels_list):
    key = 'IN_AT' if lv in ('IN', 'AT') else lv
    X_lv[i, LV_ORDER.index(key)] = 1.0

# ===== 5折CV =====
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
    gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
    gb.fit(sc.transform(X_tr), y[tr] - boosts[tr])
    oof[te] = gb.predict(sc.transform(X_te)) + boosts[te]
    print(f'  fold{fi} 完成', flush=True)
print(f'社区模型 CV: MAE={mean_absolute_error(y, oof):.4f} bias={(oof-y).mean():+.4f} r2={r2_score(y, oof):.4f}')

# ===== 全量训练 =====
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
gb_final = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
gb_final.fit(scaler.transform(X_all), y - boosts)
model = {'gb': gb_final, 'scaler': scaler, 'feature_names': gb_feature_names,
         'p95_vals': p95, 'p99_vals': p99, 'lv_order': LV_ORDER,
         'version': 'v12.6-community', 'n_train': n, 'MANUAL_FLAT': FLAT}
path = os.path.join(_ROOT, 'models', '6dim_model_v12_community.pkl')
with open(path, 'wb') as f:
    pickle.dump(model, f)
print(f'已保存: {path}')

# ===== 预测官谱对照 =====
song_difficulties = load_difficulty_tsv(os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv'))
chart_files = find_chart_files(os.path.join(_ROOT, 'data', 'chart'))
errs_by_lv = {}
errs = []
seg_lo = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    for lv in ['EZ', 'HD', 'IN', 'AT']:
        if lv not in info['levels'] or lv not in diffs: continue
        try:
            cd = load_chart_json(info['levels'][lv])
            feats = extract_features(cd)
            if not feats: continue
            key = 'IN_AT' if lv in ('IN', 'AT') else lv
            vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(key)] = 1.0
            x = np.array([[feats.get(nm, 0) for nm in gb_feature_names] + vec])
            bst = compute_boost(feats, p95, p99)
            pred = float(gb_final.predict(scaler.transform(x))[0]) + bst
            e = pred - diffs[lv]
            errs.append(e); errs_by_lv.setdefault(lv, []).append(e)
            seg_lo.append(diffs[lv])
        except Exception:
            pass
errs = np.array(errs); seg_lo = np.array(seg_lo)
print()
print('===== 社区模型 → 官谱对照 =====')
print(f'n={len(errs)} 整体bias={errs.mean():+.4f} MAE={np.abs(errs).mean():.3f}')
for lv in ['EZ', 'HD', 'IN', 'AT']:
    if errs_by_lv.get(lv):
        e = np.array(errs_by_lv[lv])
        print(f'  {lv}: n={len(e)} bias={e.mean():+.4f} MAE={np.abs(e).mean():.3f}')
for lo, hi, tag in [(11,13,'11-13'),(13,14,'13-14'),(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,99,'16.5+')]:
    m = np.where((seg_lo >= lo) & (seg_lo < hi))[0]
    if len(m):
        print(f'  定数[{tag}]: n={len(m)} bias={errs[m].mean():+.4f}')
# 特征重要性 top15 (社区模型学的权重 vs 官谱模型的差异见特征重要性对比)
imp = sorted(zip(gb_feature_names, gb_final.feature_importances_), key=lambda x: -x[1])[:20]
print()
print('社区模型特征重要性 top20:')
for nm, v in imp:
    print('  %-32s %.4f' % (nm, v))
