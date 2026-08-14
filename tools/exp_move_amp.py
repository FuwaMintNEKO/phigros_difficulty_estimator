# -*- coding: utf-8 -*-
"""线移幅度特征实验: 单事件位移/屏宽比 (大甩线特征)
假设: jline_movement_density 只统计事件数, 忽略幅度。Bonus Time 事件少但单次甩线达0.7屏宽。
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
from boost_config import MANUAL_FLAT

CACHE = os.path.join(ROOT, 'data', 'phira', '_feats_cache.npz')
TAIL_CACHE = os.path.join(ROOT, 'data', 'phira', '_tail_cache.npz')
AMP_CACHE = os.path.join(ROOT, 'data', 'phira', '_amp_cache.npz')
AMP_NAMES = ['line_move_amp_p95', 'line_move_amp_per_sec', 'line_move_amp_max',
             'line_move_amp_mean', 'line_rotate_amp_per_sec', 'line_rotate_amp_p95']

d = np.load(CACHE, allow_pickle=True)
feats_list, labels, levels_list, names_list = (d['feats_list'], d['labels'],
                                               d['levels_list'], d['names_list'])
gb_feature_names = list(d['gb_feature_names'])
n = len(feats_list)
td = np.load(TAIL_CACHE, allow_pickle=True)
TAIL_NAMES = ['tail_note_count', 'tail_ratio', 'tail_peak_1s_ratio',
              'tail_peak_vs_mean', 'tail_density', 'tail_core_share']
X_tail = np.column_stack([td[k] for k in TAIL_NAMES])
print(f'样本: {n}')

# ===== 线移幅度特征 (全量官谱重提取, 缓存) =====
def compute_amp_feats(cd, dur_sec):
    jl = cd.get('judgeLineList', [])
    amps, rots, xs = [], [], []
    for line in jl:
        for ev in line.get('judgeLineMoveEvents', []):
            sx, ex = ev.get('start', 0), ev.get('end', 0)
            sy, ey = ev.get('start2', 0), ev.get('end2', 0)
            amps.append(abs(ex - sx) + abs(ey - sy)); xs += [sx, ex]
        for ev in line.get('judgeLineRotateEvents', []):
            rots.append(abs(ev.get('end', 0) - ev.get('start', 0)))
        for layer in line.get('eventLayers', []):
            if not layer: continue
            for ev in layer.get('moveXEvents', []):
                amps.append(abs(ev.get('end', 0) - ev.get('start', 0))); xs += [ev.get('start', 0), ev.get('end', 0)]
            for ev in layer.get('moveYEvents', []):
                amps.append(abs(ev.get('end', 0) - ev.get('start', 0)))
            for ev in layer.get('rotateEvents', []):
                rots.append(abs(ev.get('end', 0) - ev.get('start', 0)))
        for ev in line.get('extended', {}).get('inclineEvents', []):
            amps.append(abs(ev.get('end', 0) - ev.get('start', 0)))
    out = {k: 0.0 for k in AMP_NAMES}
    if not amps or dur_sec <= 0:
        return out
    a = np.array(amps)
    xs = np.array(xs) if xs else np.array([0.0])
    w = max(float(np.percentile(xs, 99) - np.percentile(xs, 1)), 1e-6)
    r = a / w  # 屏宽比
    out['line_move_amp_p95'] = float(np.percentile(r, 95))
    out['line_move_amp_max'] = float(r.max())
    out['line_move_amp_mean'] = float(r.mean())
    out['line_move_amp_per_sec'] = float(r.sum() / dur_sec)
    if rots:
        ro = np.array(rots)
        rw = max(float(np.percentile(ro, 99) - np.percentile(ro, 1)), 1e-6)
        rr = ro / rw
        out['line_rotate_amp_per_sec'] = float(rr.sum() / dur_sec)
        out['line_rotate_amp_p95'] = float(np.percentile(rr, 95))
    return out

if os.path.exists(AMP_CACHE):
    print('加载线移幅度缓存...')
    ad = np.load(AMP_CACHE, allow_pickle=True)
    amp_cols = {k: ad[k] for k in AMP_NAMES}
else:
    t0 = time.time()
    song_difficulties = load_difficulty_tsv(os.path.join(ROOT, 'data', 'info', 'difficulty.tsv'))
    chart_files = find_chart_files(os.path.join(ROOT, 'data', 'chart'))
    dur_by_key = {(names_list[i], levels_list[i]): feats_list[i].get('duration_sec', 120)
                  for i in range(n)}
    amp_map = {}
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
                    amp_map[(fn, lv)] = compute_amp_feats(cd, dur)
                except Exception:
                    amp_map[(fn, lv)] = {}
    amp_cols = {k: np.zeros(n) for k in AMP_NAMES}
    for i in range(n):
        af = amp_map.get((names_list[i], levels_list[i]), {})
        for k in AMP_NAMES:
            amp_cols[k][i] = af.get(k, 0)
    np.savez(AMP_CACHE, **amp_cols)
    print(f'线移幅度特征缓存已保存 ({time.time()-t0:.1f}s)')

X_amp = np.column_stack([amp_cols[k] for k in AMP_NAMES])
for k in AMP_NAMES:
    print(f'  {k}: 范围[{amp_cols[k].min():.4f}, {amp_cols[k].max():.4f}] 非零={int((amp_cols[k]>0).sum())}')

y = np.array(labels)
LV_ORDER = ['EZ', 'HD', 'IN', 'AT']
orig_names = list(gb_feature_names)
gb_feature_names_full = orig_names + TAIL_NAMES + AMP_NAMES
X_base = np.hstack([np.array([[f.get(nn, 0) for nn in orig_names] for f in feats_list]), X_tail, X_amp])
X_lv = np.zeros((n, 4))
for i, lv in enumerate(levels_list):
    X_lv[i, LV_ORDER.index(lv)] = 1.0
levels_arr = np.array(levels_list)
groups = np.array([fn for fn in names_list])
CAPS = {'_default': 4.0}

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

def run_cv(splits, use_amp=False, tag=''):
    gn = gb_feature_names_full if use_amp else orig_names + TAIL_NAMES
    xb = X_base if use_amp else X_base[:, :len(orig_names) + len(TAIL_NAMES)]
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
splits = list(gkf.split(X_base, y, groups))

oof_base = run_cv(splits, use_amp=False, tag='基线(+尾杀): ')
oof_amp = run_cv(splits, use_amp=True, tag='+线移幅度: ')

# 相关性: 幅度特征 vs 残差改善
resid = y - oof_base
for k in AMP_NAMES:
    c = amp_cols[k]
    r = np.corrcoef(c, resid)[0, 1]
    print(f'  {k}: 与基线残差相关 r={r:+.3f}')
