# -*- coding: utf-8 -*-
"""v12.7路线A: 大社区模型(3280首高质量社区谱) + 官谱982当测试集调偏移
标签=社区定数; 官谱只用于拟合偏移与评估, 不进训练
用法: python train/train_v12_comm_big.py
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

# ===== 社区训练集: rating>=0.9 & count>=30 & 非整数 & 11<=diff<=19.5 =====
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
print(f'社区训练集: {len(sel)} 首')
JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star')
feats_list, labels, levels_list, names_list = [], [], [], []
for row in sel:
    p = os.path.join(JSON_DIR, row['id'] + '.json')
    if not os.path.exists(p):
        continue
    try:
        with open(p, 'rb') as f:
            cd, raw = load_chart_from_bytes(f.read())
        feats = extract_features(cd, speed=1.0)
        if feats:
            lv = 'AT' if 'AT' in (row['level'] or '').upper() else ('IN' if 'IN' in (row['level'] or '').upper() else 'HD')
            feats_list.append(feats); labels.append(float(row['difficulty']))
            levels_list.append(lv); names_list.append(row['id'])
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
    gb.fit(sc.transform(X_tr), y[tr] - boosts[tr])
    oof[te] = gb.predict(sc.transform(X_te)) + boosts[te]
    print(f'  fold{fi} 完成', flush=True)
print(f'社区模型 CV: MAE={mean_absolute_error(y, oof):.4f} bias={(oof-y).mean():+.4f} r2={r2_score(y, oof):.4f}')

# ===== 全量 =====
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
gb_final.fit(scaler.transform(X_all), y - boosts)
model = {'gb': gb_final, 'scaler': scaler, 'feature_names': gb_feature_names,
         'p95_vals': p95, 'p99_vals': p99, 'lv_order': LV_ORDER,
         'version': 'v12.7-comm-big', 'n_train': n, 'MANUAL_FLAT': FLAT}
path = os.path.join(_ROOT, 'models', '6dim_model_v12_comm_big.pkl')
with open(path, 'wb') as f:
    pickle.dump(model, f)
print(f'已保存: {path}')

def pred_gbboost(feats, lv):
    key = 'IN_AT' if lv in ('IN', 'AT') else lv
    if key not in LV_ORDER: key = LV_ORDER[-1]
    vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(key)] = 1.0
    x = np.array([[feats.get(nm, 0) for nm in gb_feature_names] + vec])
    bst = compute_boost(feats, p95, p99)
    return float(gb_final.predict(scaler.transform(x))[0]) + bst

# ===== 官谱982当测试集 =====
song_difficulties = load_difficulty_tsv(os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv'))
chart_files = find_chart_files(os.path.join(_ROOT, 'data', 'chart'))
off_preds, off_diffs, off_lvs = [], [], []
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
            off_preds.append(pred_gbboost(feats, lv))
            off_diffs.append(diffs[lv]); off_lvs.append(lv)
        except Exception:
            pass
off_preds = np.array(off_preds); off_diffs = np.array(off_diffs); off_lvs = np.array(off_lvs)
errs = off_preds - off_diffs
print()
print('===== 大社区模型 → 官谱(无偏移) =====')
print(f'n={len(errs)} bias={errs.mean():+.4f} MAE={np.abs(errs).mean():.3f}')
for lv in ['EZ', 'HD', 'IN', 'AT']:
    m = off_lvs == lv
    if m.sum(): print(f'  {lv}: n={m.sum()} bias={errs[m].mean():+.4f} MAE={np.abs(errs[m]).mean():.3f}')
for lo, hi, tag in [(11,13,'11-13'),(13,14,'13-14'),(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,99,'16.5+')]:
    m = (off_diffs >= lo) & (off_diffs < hi)
    if m.sum(): print(f'  定数[{tag}]: n={m.sum()} bias={errs[m].mean():+.4f}')

# ===== 偏移方案1: 等级线性偏移 (按EZ/HD/IN/AT各减bias) =====
print()
print('===== 偏移方案对比 =====')
lv_shift = {lv: errs[off_lvs == lv].mean() for lv in ['EZ','HD','IN','AT']}
e1 = errs - np.array([lv_shift[lv] for lv in off_lvs])
print(f'方案1(等级偏移): MAE={np.abs(e1).mean():.3f} bias={e1.mean():+.4f}')

# ===== 偏移方案2: 段偏移 (11-13/13-14/14-15/15-16/16-16.5/16.5+) =====
segs = [(11,13),(13,14),(14,15),(15,16),(16,16.5),(16.5,99)]
seg_shift = {}
e2 = errs.copy()
for lo, hi in segs:
    m = (off_diffs >= lo) & (off_diffs < hi)
    if m.sum():
        s = errs[m].mean()
        seg_shift[(lo,hi)] = s
        e2[m] = errs[m] - s
print(f'方案2(段偏移): MAE={np.abs(e2).mean():.3f} bias={e2.mean():+.4f} 偏移={ {k: round(v,3) for k,v in seg_shift.items()} }')

# ===== 偏移方案3: 线性回归 (pred→官谱) =====
from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(off_preds.reshape(-1,1), off_diffs)
e3 = lr.predict(off_preds.reshape(-1,1)) - off_diffs
print(f'方案3(线性回归): a={lr.coef_[0]:.4f} b={lr.intercept_:.4f} MAE={np.abs(e3).mean():.3f} bias={e3.mean():+.4f}')

# ===== 方案2偏移 + test_charts 锚点 =====
import glob
print()
print('===== 方案2(段偏移) → test_charts 锚点 =====')
anchors = [
    ('Cheerio!', 'data/test_charts/Cheerio!(17.0).json', 17.0),
    ('恋ひ恋ふ縁', 'data/test_charts/恋ひ恋ふ縁(16.8)(1).json', 16.8),
    ('おぎゃり', 'data/test_charts/おぎゃりないざー(16.4~16.6).json', 16.3),
    ('Runengon', 'data/test_charts/Runengon(16.2~16.4).json', 16.8),
    ('FinalEndGame', 'data/test_charts/The Final EndGame(18.4).json', 18.4),
    ('FB', 'data/phira/json/47264.json', 16.7),
    ('60137', 'data/phira/json_unranked_4star/60137.json', 16.75),
    ('xodus', 'data/phira/json_unranked_4star/294.json', 17.65),
    ('Apollo', 'data/phira/json_unranked_4star/41242.json', 18.0),
    ('silly', 'data/test_charts/silly-willy-nilly(17.7)(1).json', 17.7),
    ('Waking', 'data/test_charts/Waking Shadows (feat. Eili)(17.8).json', 17.8),
    ('Xaleid', 'data/test_charts/Xaleid◆scopiX(18.2)(1).json', 18.2),
]
import importlib
import app as app_mod
importlib.reload(app_mod)
for name, path, tgt in anchors:
    with open(path, 'rb') as f:
        cd, raw = load_chart_from_bytes(f.read())
    feats = extract_features(cd)
    if not feats: continue
    is_custom = app_mod.is_custom_chart(cd, raw)
    if is_custom:
        feats = app_mod.apply_domain_align(feats, True, 'IN')
    p = pred_gbboost(feats, 'IN')
    for lo, hi in segs:
        if lo < p <= hi:
            p -= seg_shift.get((lo,hi), 0.0)
            break
    print('%-14s: %.2f (目标%.2f 差%+.2f)' % (name, p, tgt, p - tgt))
