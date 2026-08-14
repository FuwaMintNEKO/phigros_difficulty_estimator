# -*- coding: utf-8 -*-
"""boost 权重微调实验: 依据残差诊断
被低估(提高): stair_speed_avg(-0.265), fast_hold_ratio(-0.146), weighted_mf_score_per_sec(-0.114)
被高估(降低): position_entropy(+0.150), discrete_mf_ratio(+0.146), flick_ratio(+0.142), hold_duration(+0.135)
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
gb_names = orig_names + TAIL_NAMES
X_base = np.hstack([np.array([[f.get(nn, 0) for nn in orig_names] for f in feats_list]), X_tail])
X_lv = np.zeros((n, 4))
for i, lv in enumerate(levels_list):
    X_lv[i, LV_ORDER.index(lv)] = 1.0
CAPS = {'_default': 4.0}

def make_flat(overrides=None, additions=None):
    """返回调整后的 FLAT 列表"""
    d = {f: (bl, co) for f, bl, co in MANUAL_FLAT}
    for f, bl, co in (overrides or []):
        if f in d:
            d[f] = (bl, co)
    flat = [(f, bl, co) for f, (bl, co) in d.items()]
    flat += (additions or [])
    return flat

def compute_boost(feats, p95_vals, p99_vals, flat):
    total = 0.0
    cap = CAPS.get('_default', None)
    for fname, bl, co in flat:
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

def run_cv(splits, flat, tag=''):
    # 收集 flat 里所有特征名 (含新增, 用于 per-fold 分位数)
    flat_names = [f for f, _, _ in flat]
    oof = np.zeros(n)
    for tr, te in splits:
        p95_vals, p99_vals = {}, {}
        for j, name in enumerate(gb_names):
            col = X_base[tr, j]
            p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
            p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
        for fname in flat_names:
            if fname in feats_list[0]:
                col = np.array([f.get(fname, 0) for f in feats_list])[tr]
                p95_vals[fname] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
                p99_vals[fname] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
        boosts = np.array([compute_boost(f, p95_vals, p99_vals, flat) for f in feats_list])
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

base_flat = make_flat()
run_cv(splits, base_flat, tag='[基线] ')

# stair_speed_avg coef 扫描
for co in [0.24, 0.28, 0.34, 0.40]:
    run_cv(splits, make_flat(overrides=[('stair_speed_avg', 8.0, co)]),
           tag=f'[stair_speed_avg co={co}] ')

# 降低被高估的 position_entropy
run_cv(splits, make_flat(overrides=[('position_entropy', 2.0, 0.030)]),
       tag='[position_entropy co=0.030] ')

# 组合: stair_speed_avg 提高 + position_entropy 降低
run_cv(splits, make_flat(overrides=[('stair_speed_avg', 8.0, 0.28), ('position_entropy', 2.0, 0.030)]),
       tag='[stair↑ + posent↓] ')
