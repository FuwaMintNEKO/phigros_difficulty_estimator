# -*- coding: utf-8 -*-
"""实验: 合并 IN/AT level onehot, 看分组CV是否改善
对比: 4类(EZ/HD/IN/AT) vs 3类(EZ/HD/IN_AT合并) vs 2类(EZ/HD vs IN_AT)
"""
import os, sys, io, pickle
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
levels_arr = np.array(levels_list)
groups = np.array([fn for fn in names_list])

# 先看 level_IN/AT 的重要性
m = pickle.load(open(os.path.join(ROOT, 'models', '6dim_model_v10.pkl'), 'rb'))
imp = m['gb'].feature_importances_
fn = list(m['feature_names']) + list(m['lv_order'])
for i, name in enumerate(fn):
    if name.startswith('level'):
        print(f'  {name} 重要性 = {imp[i]:.5f}')

gb_names = orig_names + TAIL_NAMES
X_base = np.hstack([np.array([[f.get(nn, 0) for nn in orig_names] for f in feats_list]), X_tail])
CAPS = {'_default': 4.0}

def compute_boost(feats, p95_vals, p99_vals):
    total = 0.0
    cap = CAPS.get('_default', None)
    for fname, bl, co in MANUAL_FLAT:
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

def build_level(levels, mode):
    if mode == '4class':
        X_lv = np.zeros((len(levels), 4))
        for i, lv in enumerate(levels):
            X_lv[i, ['EZ','HD','IN','AT'].index(lv)] = 1.0
    elif mode == '3class':  # IN/AT 合并
        X_lv = np.zeros((len(levels), 3))
        for i, lv in enumerate(levels):
            if lv == 'EZ': X_lv[i, 0] = 1.0
            elif lv == 'HD': X_lv[i, 1] = 1.0
            else: X_lv[i, 2] = 1.0
    elif mode == '2class':  # EZ/HD 合并, IN/AT 合并
        X_lv = np.zeros((len(levels), 2))
        for i, lv in enumerate(levels):
            if lv in ('EZ', 'HD'): X_lv[i, 0] = 1.0
            else: X_lv[i, 1] = 1.0
    elif mode == 'nolevel':  # 完全无 level
        X_lv = np.zeros((len(levels), 1))
    return X_lv

def run_cv(splits, X_lv, tag=''):
    oof = np.zeros(n)
    for tr, te in splits:
        p95_vals, p99_vals = {}, {}
        for j, name in enumerate(gb_names):
            col = X_base[tr, j]
            p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
            p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
        boosts = np.array([compute_boost(f, p95_vals, p99_vals) for f in feats_list])
        X_tr = np.hstack([X_base[tr], X_lv[tr]])
        X_te = np.hstack([X_base[te], X_lv[te]])
        sc = StandardScaler().fit(X_tr)
        gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                       learning_rate=0.05, subsample=0.8, random_state=42)
        gb.fit(sc.transform(X_tr), y[tr] - boosts[tr])
        oof[te] = gb.predict(sc.transform(X_te)) + boosts[te]
    print(f'{tag}整体MAE = {mean_absolute_error(y, oof):.4f}', end='')
    for lv in ['EZ', 'HD', 'IN', 'AT']:
        mm = np.where(levels_arr == lv)[0]
        if len(mm):
            print(f'  {lv}={mean_absolute_error(y[mm], oof[mm]):.4f}', end='')
    print()
    return oof

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_base, y, groups))

run_cv(splits, build_level(levels_arr, '4class'), tag='[4类 EZ/HD/IN/AT] ')
run_cv(splits, build_level(levels_arr, '3class'), tag='[3类 EZ/HD/IN+AT] ')
run_cv(splits, build_level(levels_arr, '2class'), tag='[2类 EZHD/INAT] ')
run_cv(splits, build_level(levels_arr, 'nolevel'), tag='[无level] ')
