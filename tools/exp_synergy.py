# -*- coding: utf-8 -*-
"""协同/位移维度实验: 验证 (1) 位移密度 movement_per_second 进 boost
(2) 交叉手密度 cross_hand_density 进 boost (3) 协同特征 (多维度高位乘积)
框架复用 _feats_cache.npz + 歌曲分组5折CV
"""
import os, sys, io
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator'
sys.path.insert(0, ROOT)
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from boost_config import MANUAL_FLAT

d = np.load(os.path.join(ROOT, 'data', 'phira', '_feats_cache.npz'), allow_pickle=True)
feats_list = d['feats_list']; labels = d['labels']
levels_list = d['levels_list']; names_list = d['names_list']
orig_names = list(d['gb_feature_names'])
TAIL_NAMES = ['tail_note_count', 'tail_ratio', 'tail_peak_1s_ratio',
              'tail_peak_vs_mean', 'tail_density', 'tail_core_share']
td = np.load(os.path.join(ROOT, 'data', 'phira', '_tail_cache.npz'), allow_pickle=True)
X_tail = np.column_stack([td[k] for k in TAIL_NAMES])

n = len(feats_list)
y = np.array(labels)
LV_ORDER = ['EZ', 'HD', 'IN', 'AT']
levels_arr = np.array(levels_list)
groups = np.array([fn for fn in names_list])

# 检查目标字段是否在缓存中
for fname in ['movement_per_second', 'cross_hand_density', 'multi_line_sim_events',
              'multi_finger_3plus_events', 'real_core_notes_per_second', 'bpm']:
    print(f'{fname}: {"有" if fname in feats_list[0] else "缺"}')

gb_names = orig_names + TAIL_NAMES
X_base = np.hstack([np.array([[f.get(nn, 0) for nn in orig_names] for f in feats_list]), X_tail])
X_lv = np.zeros((n, 4))
for i, lv in enumerate(levels_list):
    X_lv[i, LV_ORDER.index(lv)] = 1.0
CAPS = {'_default': 4.0}

def compute_boost(feats, p95_vals, p99_vals, extra_flat=None):
    total = 0.0
    cap = CAPS.get('_default', None)
    flat = MANUAL_FLAT + (extra_flat or [])
    for fname, bl, co in flat:
        v = feats.get(fname, 0)
        pv = p95_vals.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t:
            continue
        e = v / t - 1.0
        c = CAPS.get(fname, cap)
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

def run_cv(splits, tag='', extra_flat=None):
    oof = np.zeros(n)
    for fi, (tr, te) in enumerate(splits):
        p95_vals, p99_vals = {}, {}
        for j, name in enumerate(gb_names):
            col = X_base[tr, j]
            p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
            p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
        # 额外特征的分位数 (per-fold)
        if extra_flat:
            for fname, bl, co in extra_flat:
                if fname not in feats_list[0]:
                    continue
                col = np.array([f.get(fname, 0) for f in feats_list])[tr]
                p95_vals[fname] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
                p99_vals[fname] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
        boosts = np.array([compute_boost(f, p95_vals, p99_vals, extra_flat) for f in feats_list])
        X_tr = np.hstack([X_base[tr], X_lv[tr]])
        X_te = np.hstack([X_base[te], X_lv[te]])
        sc = StandardScaler().fit(X_tr)
        gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                       learning_rate=0.05, subsample=0.8, random_state=42)
        gb.fit(sc.transform(X_tr), y[tr] - boosts[tr])
        oof[te] = gb.predict(sc.transform(X_te)) + boosts[te]
    print(f'{tag}整体MAE = {mean_absolute_error(y, oof):.4f}', end='')
    for lv in LV_ORDER:
        m = np.where(levels_arr == lv)[0]
        print(f'  {lv}={mean_absolute_error(y[m], oof[m]):.4f}', end='')
    print()
    return oof

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_base, y, groups))

# 基线
oof0 = run_cv(splits, tag='[基线] ')

# 位移密度进 boost (扫描 baseline/coef)
for bl, co in [(4, 0.05), (6, 0.08), (8, 0.10), (8, 0.15), (10, 0.12)]:
    run_cv(splits, tag=f'[+movement_per_sec bl={bl} co={co}] ', extra_flat=[('movement_per_second', bl, co)])

# 交叉手密度进 boost
for bl, co in [(1.0, 0.10), (1.5, 0.15), (2.0, 0.20)]:
    run_cv(splits, tag=f'[+cross_hand_density bl={bl} co={co}] ', extra_flat=[('cross_hand_density', bl, co)])

# 两者一起
run_cv(splits, tag='[+movement +cross_hand] ',
       extra_flat=[('movement_per_second', 8, 0.10), ('cross_hand_density', 1.5, 0.15)])
