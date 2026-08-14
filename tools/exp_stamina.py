# -*- coding: utf-8 -*-
"""底力维度诊断: 验证"底力 = BPM × 密度"是否被模型遗漏。
用户模型: 难度 = 配置强度(多指/位移/多面, 加分) + 底力消耗(BPM×密度, 加分)
ALL☆NIGHTER!: 配置强(加分) 但 BPM低(底力减分) → 应17.0
BonusTime: 配置(多面) + BPM高(底力加分) → 应16.5
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

def g(f):
    return np.array([x.get(f, 0) for x in feats_list])

BPM = g('bpm')
CORE_NPS = g('real_core_notes_per_second')
TAP_NPS = g('tap_notes_per_second')
NPS = g('real_notes_per_second')
CORE_PER_BEAT = g('core_notes_per_beat')
MF = g('weighted_mf_score_total')
MULTILINE = g('multi_line_sim_events')
MOVEMENT = g('movement_per_second')

# 底力/协调候选特征
cands = {
    'bpm_x_core_nps': BPM * CORE_NPS,                       # 底力 = BPM×核心密度
    'bpm_x_tap_nps': BPM * TAP_NPS,
    'bpm_x_nps': BPM * NPS,
    'core_nps_per_bpm': CORE_NPS / (BPM + 1.0),             # 协调型 = 密度相对BPM (低BPM高密度时大)
    'core_per_beat': CORE_PER_BEAT,                          # 每拍密度 (协调)
    'core_per_beat_x_bpm': CORE_PER_BEAT * BPM,             # 底力 = 每拍密度×BPM
    'bpm_sq': BPM ** 2,
    'log_bpm_x_core': np.log1p(BPM) * CORE_NPS,
}

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

def run_cv(splits, X, gn, tag=''):
    oof = np.zeros(n)
    for tr, te in splits:
        p95_vals, p99_vals = {}, {}
        for j, name in enumerate(gn):
            col = X[tr, j]
            p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
            p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
        boosts = np.array([compute_boost(f, p95_vals, p99_vals) for f in feats_list])
        X_tr = np.hstack([X[tr], X_lv[tr]]); X_te = np.hstack([X[te], X_lv[te]])
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
oof0 = run_cv(splits, X_base, gb_names, tag='[基线] ')
resid = y - oof0
mask = (levels_arr == 'IN') | (levels_arr == 'AT')

print('\n===== 底力/协调候选 与残差相关 (IN/AT) =====')
for name, col in cands.items():
    c = col[mask]
    if np.std(c) < 1e-9: continue
    r = np.corrcoef(c, resid[mask])[0, 1]
    print(f'  {name:<24} 与残差相关={r:+.3f}  (负=该维度越高越被低估)')

# 进 GB 测试 (有信号才值得)
print('\n===== 底力特征进 GB 测试 =====')
for name, col in cands.items():
    X_new = np.hstack([X_base, col.reshape(-1, 1)])
    run_cv(splits, X_new, gb_names + [name], tag=f'[+{name}] ')
