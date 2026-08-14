# -*- coding: utf-8 -*-
"""组合实验: 线移幅度(log) + 同时活跃线数 + 协同分数(修正维度)
目标: 捕捉"大甩线位移/多面下落/多维协同"三个盲区
"""
import os, sys, io, time
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator'
sys.path.insert(0, ROOT)
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import collect_all_notes, time_to_seconds
from boost_config import MANUAL_FLAT

CACHE = os.path.join(ROOT, 'data', 'phira', '_feats_cache.npz')
TAIL_CACHE = os.path.join(ROOT, 'data', 'phira', '_tail_cache.npz')
AMP_CACHE = os.path.join(ROOT, 'data', 'phira', '_amp_cache.npz')
AL_CACHE = os.path.join(ROOT, 'data', 'phira', '_active_lines_cache.npz')

d = np.load(CACHE, allow_pickle=True)
feats_list, labels, levels_list, names_list = (d['feats_list'], d['labels'],
                                               d['levels_list'], d['names_list'])
gb_feature_names = list(d['gb_feature_names'])
n = len(feats_list)
td = np.load(TAIL_CACHE, allow_pickle=True)
TAIL_NAMES = ['tail_note_count', 'tail_ratio', 'tail_peak_1s_ratio',
              'tail_peak_vs_mean', 'tail_density', 'tail_core_share']
X_tail = np.column_stack([td[k] for k in TAIL_NAMES])
ad = np.load(AMP_CACHE, allow_pickle=True)
AMP_NAMES = ['line_move_amp_p95', 'line_move_amp_per_sec', 'line_move_amp_max',
             'line_move_amp_mean', 'line_rotate_amp_per_sec', 'line_rotate_amp_p95']
amp_raw = {k: ad[k] for k in AMP_NAMES}
print(f'样本: {n}')

# ===== 同时活跃线数 (1秒bin内有多少条线有note) =====
AL_NAMES = ['active_line_max', 'active_line_mean', 'active_line_heavy_ratio']
def compute_al(cd, dur_sec):
    all_notes, judge_lines, bpm_tl = collect_all_notes(cd)
    if not all_notes:
        return {k: 0.0 for k in AL_NAMES}
    fb = judge_lines[0].get('bpm', 120.0) if judge_lines else 120.0
    nb = max(int(np.ceil(dur_sec)), 1)
    bins = [set() for _ in range(nb)]
    for nd in all_notes:
        sec = time_to_seconds(nd['time'], max(nd.get('bpm', fb), 1.0), bpm_tl)
        idx = int(sec)
        if 0 <= idx < nb:
            bins[idx].add(nd.get('judge_line_idx', 0))
    counts = np.array([len(b) for b in bins])
    if counts.size == 0:
        return {k: 0.0 for k in AL_NAMES}
    return {'active_line_max': float(counts.max()),
            'active_line_mean': float(counts.mean()),
            'active_line_heavy_ratio': float((counts >= 4).sum() / counts.size)}

if os.path.exists(AL_CACHE):
    print('加载活跃线缓存...')
    ald = np.load(AL_CACHE, allow_pickle=True)
    al_cols = {k: ald[k] for k in AL_NAMES}
else:
    t0 = time.time()
    song_difficulties = load_difficulty_tsv(os.path.join(ROOT, 'data', 'info', 'difficulty.tsv'))
    chart_files = find_chart_files(os.path.join(ROOT, 'data', 'chart'))
    dur_by_key = {(names_list[i], levels_list[i]): feats_list[i].get('duration_sec', 120)
                  for i in range(n)}
    al_map = {}
    for fn, info in chart_files.items():
        sid = info['song_id']
        if sid not in song_difficulties:
            continue
        diffs = song_difficulties[sid]
        for lv in ['EZ', 'HD', 'IN', 'AT']:
            if lv in info['levels'] and lv in diffs:
                try:
                    cd = load_chart_json(info['levels'][lv])
                    dur = dur_by_key.get((fn, lv), 120.0)
                    al_map[(fn, lv)] = compute_al(cd, dur)
                except Exception:
                    al_map[(fn, lv)] = {}
    al_cols = {k: np.zeros(n) for k in AL_NAMES}
    for i in range(n):
        af = al_map.get((names_list[i], levels_list[i]), {})
        for k in AL_NAMES:
            al_cols[k][i] = af.get(k, 0)
    np.savez(AL_CACHE, **al_cols)
    print(f'活跃线缓存已保存 ({time.time()-t0:.1f}s)')

X_al = np.column_stack([al_cols[k] for k in AL_NAMES])
for k in AL_NAMES:
    print(f'  {k}: 范围[{al_cols[k].min():.2f}, {al_cols[k].max():.2f}]')

# ===== 幅度特征: log1p + winsorize P99 =====
def prep_amp(k):
    v = amp_raw[k]
    p99 = np.percentile(v, 99)
    v = np.clip(v, 0, p99)
    return np.log1p(v)
X_amp = np.column_stack([prep_amp(k) for k in AMP_NAMES])

y = np.array(labels)
LV_ORDER = ['EZ', 'HD', 'IN', 'AT']
orig_names = list(gb_feature_names)
X_gb = np.array([[f.get(nn, 0) for nn in orig_names] for f in feats_list])
X_lv = np.zeros((n, 4))
for i, lv in enumerate(levels_list):
    X_lv[i, LV_ORDER.index(lv)] = 1.0
levels_arr = np.array(levels_list)
groups = np.array([fn for fn in names_list])
CAPS = {'_default': 4.0}

# ===== 协同分数 (修正维度) =====
dim_defs = [
    ('density', 'core_peak_density_1sec_top5avg'),
    ('read', 'jline_movement_density'),
    ('coord', 'sim_pos_spread_max'),
    ('stamina', 'duration_sec'),
]
dims_pct = {}
for dim, fn in dim_defs:
    col = X_gb[:, orig_names.index(fn)]
    dims_pct[dim] = (col[:, None] <= col).mean(axis=1)
syn_min = np.min(np.column_stack(list(dims_pct.values())), axis=1)
syn_cnt = np.sum(np.column_stack([p > 0.7 for p in dims_pct.values()]), axis=1).astype(float)

def compute_boost(feats, p95_vals, p99_vals):
    total = 0.0
    for fname, bl, co in MANUAL_FLAT:
        v = feats.get(fname, 0)
        t = max(p95_vals.get(fname, 0) * 0.55, bl * 0.5)
        if v <= t:
            continue
        e = v / t - 1.0
        c = CAPS.get(fname, CAPS.get('_default', None))
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

def run_cv(splits, add_cols=None, add_names=(), tag=''):
    xb = np.hstack([X_gb, X_tail]) if add_cols is None else np.hstack([X_gb, X_tail] + add_cols)
    gn = orig_names + TAIL_NAMES + list(add_names)
    oof = np.zeros(n)
    for fi, (tr, te) in enumerate(splits):
        p95_vals, p99_vals = {}, {}
        for j, name in enumerate(gn):
            col = xb[tr, j]
            p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
            p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
        boosts = np.array([compute_boost(f, p95_vals, p99_vals) for f in feats_list])
        sc = StandardScaler().fit(np.hstack([xb[tr], X_lv[tr]]))
        gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                       learning_rate=0.05, subsample=0.8, random_state=42)
        gb.fit(sc.transform(np.hstack([xb[tr], X_lv[tr]])), y[tr] - boosts[tr])
        oof[te] = gb.predict(sc.transform(np.hstack([xb[te], X_lv[te]]))) + boosts[te]
    mae = mean_absolute_error(y, oof)
    print(f'{tag}整体MAE = {mae:.4f}', end='')
    for lv in LV_ORDER:
        m = np.where(levels_arr == lv)[0]
        print(f'  {lv}={mean_absolute_error(y[m], oof[m]):.4f}', end='')
    print()
    return oof

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_gb, y, groups))

oof_base = run_cv(splits, tag='基线(+尾杀): ')
oof_amp = run_cv(splits, add_cols=[X_amp], add_names=AMP_NAMES, tag='+幅度(log): ')
oof_al = run_cv(splits, add_cols=[X_al], add_names=AL_NAMES, tag='+活跃线: ')
oof_syn = run_cv(splits, add_cols=[syn_min[:, None], syn_cnt[:, None]],
                 add_names=['syn_min', 'syn_cnt'], tag='+协同: ')
oof_all = run_cv(splits, add_cols=[X_amp, X_al, syn_min[:, None], syn_cnt[:, None]],
                 add_names=AMP_NAMES + AL_NAMES + ['syn_min', 'syn_cnt'], tag='+全部: ')

# 残差相关
resid = y - oof_base
for name, arr in [('line_move_amp_p95(log)', X_amp[:, 0]), ('active_line_mean', X_al[:, 1]),
                  ('syn_min', syn_min), ('syn_cnt', syn_cnt)]:
    print(f'  {name}: 与基线残差相关 r={np.corrcoef(arr, resid)[0, 1]:+.3f}')
