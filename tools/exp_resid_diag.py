# -*- coding: utf-8 -*-
"""残差诊断: 找出模型没学好的维度。
残差 = 真实定数 - OOF预测。某特征与残差负相关显著 → 该特征高的谱被系统性低估(权重不足)。
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

def run_cv(splits):
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
    return oof

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_base, y, groups))
oof = run_cv(splits)
print(f'基线整体MAE = {mean_absolute_error(y, oof):.4f}')
resid = y - oof

# 只看 IN/AT (高难谱, 排除低段噪声)
mask = (levels_arr == 'IN') | (levels_arr == 'AT')
resid_ia = resid[mask]

# 对所有特征算残差相关性 (含被GB排除的特征, 它们可能正是遗漏维度)
all_feat_names = sorted(feats_list[0].keys())
print('\n===== 与残差相关性最强的特征 (IN/AT, |r| 降序, 只看前30) =====')
rows = []
for fn in all_feat_names:
    col = np.array([f.get(fn, 0) for f in feats_list])[mask]
    if np.std(col) < 1e-9:
        continue
    r = np.corrcoef(col, resid_ia)[0, 1]
    if not np.isnan(r):
        rows.append((fn, r, np.std(col)))
rows.sort(key=lambda x: -abs(x[1]))
print(f'{"特征":<36}{"与残差相关r":>12}{"标准差":>10}')
for fn, r, sd in rows[:30]:
    flag = '←低估(特征高但预测低)' if r < -0.05 else ('←高估' if r > 0.05 else '')
    print(f'  {fn:<36}{r:>+12.3f}{sd:>10.1f}  {flag}')

# 特别关注: 多面下落 / 位移 / BPM 相关特征
print('\n===== 重点维度残差相关性 =====')
for kw in ['multi_line', 'movement', 'cross_hand', 'bpm', 'position', 'judge_line', 'sim_']:
    for fn, r, sd in rows:
        if kw in fn:
            print(f'  {fn:<36}{r:>+12.3f}')
            break
