# -*- coding: utf-8 -*-
"""v12: 官谱982 + unranked高共识1000 混合训练
unranked标签 = 社区定数 difficulty (谱师标+rating确认)
用法: python train/train_v12_mixed.py
"""
import os, sys, io, json, csv, pickle, numpy as np
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

FLAT = MANUAL_FLAT
CAPS = {'_default': 4.0}

# ===== 1. 官谱 =====
song_difficulties = load_difficulty_tsv(os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv'))
chart_files = find_chart_files(os.path.join(_ROOT, 'data', 'chart'))
items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    for lv in ['EZ','HD','IN','AT']:
        if lv in info['levels'] and lv in diffs:
            items.append({'name': fn, 'lv': lv, 'diff': diffs[lv], 'path': info['levels'][lv], 'src': 'official'})
print(f'官谱: {len(items)}')

# ===== 2. unranked 1000 =====
train_list = []
with open(os.path.join(_ROOT, 'data', 'phira', 'train_unranked_1000.csv'), encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        train_list.append(row)
print(f'unranked训练集: {len(train_list)}')
JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star')

# ===== 提取特征 =====
feats_list, labels, levels_list, names_list = [], [], [], []
for it in items:
    try:
        cd = load_chart_json(it['path'])
        feats = extract_features(cd)
        if feats:
            feats_list.append(feats); labels.append(it['diff'])
            levels_list.append(it['lv']); names_list.append('official|' + it['name'])
    except Exception:
        pass
ok_unr = 0
for row in train_list:
    p = os.path.join(JSON_DIR, row['id'] + '.json')
    try:
        with open(p, 'rb') as f:
            cd, raw = load_chart_from_bytes(f.read())
        feats = extract_features(cd, speed=1.0)
        if feats:
            lv = 'AT' if 'AT' in (row['level'] or '').upper() else ('IN' if 'IN' in (row['level'] or '').upper() else 'HD')
            feats_list.append(feats); labels.append(float(row['diff']))
            levels_list.append(lv); names_list.append('unranked|' + str(row['id']))
            ok_unr += 1
    except Exception:
        pass
print(f'unranked特征成功: {ok_unr}')
n = len(feats_list)
print(f'总数: {n} (官谱+unranked)')

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
    if key not in LV_ORDER: key = 'IN_AT'
    X_lv[i, LV_ORDER.index(key)] = 1.0
levels_arr = np.array(levels_list)
src_arr = np.array([nm.split('|')[0] for nm in names_list])
# 加权: unranked高难段更重要 (社区标尺); 官谱EZ/HD权重低
SAMPLE_W = np.ones(n)
SAMPLE_W[src_arr == 'official'] = 1.0
SAMPLE_W[(src_arr == 'official') & ((levels_arr == 'EZ') | (levels_arr == 'HD'))] = 0.5
SAMPLE_W[(src_arr == 'unranked') & (y >= 16.5)] = 2.0
SAMPLE_W[(src_arr == 'unranked') & (y >= 17.0)] = 3.0
print(f'加权分布: 官谱EZ/HD=0.5, 官谱其他=1.0, unranked16.5+=2.0, unranked17+=3.0')

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

# ===== 歌曲分组5折CV (分组: 官谱按曲, unranked按id) =====
gkf = GroupKFold(n_splits=5)
groups = np.array([nm.rsplit('|', 1)[-1] for nm in names_list])
splits = list(gkf.split(X_base, y, groups=groups))
oof = np.zeros(n)
for fi, (tr, te) in enumerate(splits):
    p95_vals, p99_vals = {}, {}
    for j, name in enumerate(gb_feature_names):
        col = X_base[tr, j]
        p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
        p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
    boosts = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in feats_list])
    sc = StandardScaler().fit(X_base[tr])
    gb = GradientBoostingRegressor(n_estimators=600, max_depth=5, min_samples_leaf=3,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
    gb.fit(sc.transform(X_base[tr]), y[tr] - boosts[tr], sample_weight=SAMPLE_W[tr])
    oof[te] = gb.predict(sc.transform(X_base[te])) + boosts[te]
    print(f'  fold{fi} OOF完成', flush=True)

errs = oof - y
print(f'\n===== 混合CV (官谱+unranked) =====')
print(f'整体MAE = {mean_absolute_error(y, oof):.4f} | bias = {errs.mean():+.4f}')
for src in ['official', 'unranked']:
    mk = np.where(src_arr == src)[0]
    print(f'  {src}: n={len(mk)} MAE={mean_absolute_error(y[mk], oof[mk]):.4f} bias={errs[mk].mean():+.4f}')
for lo, hi, tag in [(13, 14, '13-14'), (14, 15, '14-15'), (15, 16, '15-16'), (16, 16.5, '16-16.5'), (16.5, 17, '16.5-17'), (17, 99, '17+')]:
    mk = np.where((y >= lo) & (y < hi))[0]
    if len(mk):
        print(f'  定数[{tag}]: n={len(mk)} MAE={mean_absolute_error(y[mk], oof[mk]):.4f} bias={errs[mk].mean():+.4f}')

# ===== 全量重训 + 保存 =====
p95_vals, p99_vals = {}, {}
for j, name in enumerate(gb_feature_names):
    col = X_base[:, j]
    p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
    p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
boosts = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in feats_list])
scaler = StandardScaler().fit(X_base)
gb_final = GradientBoostingRegressor(n_estimators=600, max_depth=5, min_samples_leaf=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
gb_final.fit(scaler.transform(X_base), y - boosts, sample_weight=SAMPLE_W)
model = {
    'gb': gb_final, 'scaler': scaler, 'feature_names': gb_feature_names,
    'p95_vals': p95_vals, 'p99_vals': p99_vals, 'lv_order': LV_ORDER,
    'version': 'v12-mixed-official982+unranked1000',
    'n_train': n, 'MANUAL_FLAT': FLAT, 'caps': CAPS,
    'train_meta': {'n': n, 'songs': len(set(groups)),
                   'cv_mae': float(mean_absolute_error(y, oof)),
                   'cv_r2': float(r2_score(y, oof)),
                   'official': int(np.sum(src_arr == 'official')),
                   'unranked': int(np.sum(src_arr == 'unranked'))},
}
path = os.path.join(_ROOT, 'models', '6dim_model_v12_mixed.pkl')
with open(path, 'wb') as f:
    pickle.dump(model, f)
print(f'\n已保存: {path}')