# -*- coding: utf-8 -*-
"""v12.7路线C: 预训练+微调 — 大社区模型(warm_start)用官谱982继续加树微调定标尺
评估: 5折(官谱4折微调→测1折OOF) + test_charts锚点
用法: python train/train_v12_finetune.py
"""
import os, sys, io, pickle, copy, numpy as np
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
            official_items.append({'name': fn, 'lv': lv, 'diff': diffs[lv], 'path': info['levels'][lv]})
print(f'官谱: {len(official_items)}')

feats_list, labels, levels_list, names_list = [], [], [], []
for it in official_items:
    try:
        cd = load_chart_json(it['path'])
        feats = extract_features(cd)
        if feats:
            feats_list.append(feats); labels.append(it['diff'])
            levels_list.append(it['lv']); names_list.append(it['name'])
    except Exception:
        pass
n = len(feats_list)
print(f'官谱特征: {n}')
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

# ===== 社区预训练模型 =====
comm = pickle.load(open(os.path.join(_ROOT, 'models', '6dim_model_v12_comm_big.pkl'), 'rb'))
comm_gb = comm['gb']
comm_scaler = comm['scaler']
comm_fn = comm['feature_names']
assert comm_fn == gb_feature_names, '特征列表不一致'
p95c, p99c = comm['p95_vals'], comm['p99_vals']

def comm_predict(feats, lv):
    key = 'IN_AT' if lv in ('IN', 'AT') else lv
    vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(key)] = 1.0
    x = np.array([[feats.get(nm, 0) for nm in gb_feature_names] + vec])
    bst = compute_boost(feats, p95c, p99c)
    return float(comm_gb.predict(comm_scaler.transform(x))[0]) + bst

# 微调: 每折用社区模型warm_start在官谱4折上加树(300), 测1折OOF
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_base, y, groups=np.array(names_list)))
oof = np.zeros(n)
for fi, (tr, te) in enumerate(splits):
    # 官谱侧的p95/p99 (微调时的boost用官谱分布)
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
    # v12.7: 微调=社区模型warm_start继续加树; 特征必须用社区scaler转换(与已有树的决策边界一致)
    gb = copy.deepcopy(comm_gb)
    gb.set_params(n_estimators=comm_gb.n_estimators + 300)
    gb.fit(comm_scaler.transform(X_tr), y[tr] - boosts[tr])
    oof[te] = gb.predict(comm_scaler.transform(X_te)) + boosts[te]
    print(f'  fold{fi} OOF完成', flush=True)
errs = oof - y
print(f'官谱5折微调OOF: MAE={mean_absolute_error(y, oof):.4f} bias={errs.mean():+.4f}')
for lv in ['EZ', 'HD', 'IN', 'AT']:
    m = np.array(levels_list) == lv
    print(f'  {lv}: n={m.sum()} MAE={mean_absolute_error(y[m], oof[m]):.4f} bias={errs[m].mean():+.4f}')
for lo, hi, tag in [(11,13,'11-13'),(13,14,'13-14'),(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,99,'16.5+')]:
    m = (y >= lo) & (y < hi)
    if m.sum(): print(f'  定数[{tag}]: n={m.sum()} MAE={mean_absolute_error(y[m], oof[m]):.4f} bias={errs[m].mean():+.4f}')

# ===== 全量微调 + 锚点 =====
p95, p99 = {}, {}
for j, nm in enumerate(gb_feature_names):
    col = X_base[:, j]
    p95[nm] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
    p99[nm] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
for k, v in _JLINE_P95_FIX.items():
    if k in p95: p95[k] = v
boosts = np.array([compute_boost(f, p95, p99) for f in feats_list])
X_all = np.hstack([X_base, X_lv])
gb_final = copy.deepcopy(comm_gb)
gb_final.set_params(n_estimators=comm_gb.n_estimators + 300)
gb_final.fit(comm_scaler.transform(X_all), y - boosts)
model = {'gb': gb_final, 'scaler': comm_scaler, 'feature_names': gb_feature_names,
         'p95_vals': p95, 'p99_vals': p99, 'lv_order': LV_ORDER,
         'version': 'v12.7-finetune', 'n_train': n, 'MANUAL_FLAT': FLAT}
path = os.path.join(_ROOT, 'models', '6dim_model_v12_finetune.pkl')
with open(path, 'wb') as f:
    pickle.dump(model, f)
print(f'已保存: {path}')

print()
print('===== 微调模型 → test_charts 锚点 (GB+boost, 无规则/校准) =====')
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
    ('ギザバ', 'data/test_charts/ギザバ怪文書(18.3).json', 18.3),
    ('朧月', 'data/test_charts/朧月(18.4)(1).json', 18.4),
    ('Submerged', 'data/test_charts/Submerged City(17.8).json', 17.8),
]
def finetune_pred(feats, lv):
    key = 'IN_AT' if lv in ('IN', 'AT') else lv
    vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(key)] = 1.0
    x = np.array([[feats.get(nm, 0) for nm in gb_feature_names] + vec])
    bst = compute_boost(feats, p95, p99)
    return float(gb_final.predict(comm_scaler.transform(x))[0]) + bst
errs_a = []
for name, path, tgt in anchors:
    with open(path, 'rb') as f:
        cd, raw = load_chart_from_bytes(f.read())
    feats = extract_features(cd)
    if not feats: continue
    is_custom = app_mod.is_custom_chart(cd, raw)
    if is_custom:
        feats = app_mod.apply_domain_align(feats, True, 'IN')
    p = finetune_pred(feats, 'IN')
    errs_a.append(abs(p - tgt))
    print('%-14s: %.2f (目标%.2f 差%+.2f)' % (name, p, tgt, p - tgt))
print('锚点MAE=%.2f' % (sum(errs_a)/len(errs_a)))
