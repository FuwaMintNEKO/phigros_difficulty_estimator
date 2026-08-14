# -*- coding: utf-8 -*-
"""协同特征实验 v2:
1. 基线 OOF 残差诊断: 哪些"多维度协同"组合能解释残差
2. 把协同交互特征作为新 GB 列, 跑歌曲分组 CV 对比
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

def g(fname):
    return np.array([f.get(fname, 0) for f in feats_list])

# 原始维度列
DENSITY = g('real_core_notes_per_second')
MOVEMENT = g('movement_per_second')
MULTILINE = g('multi_line_sim_events')
MF3 = g('multi_finger_3plus_events')
CROSSHAND = g('cross_hand_density')
BPM = g('bpm')
JLINE = g('judge_line_count')

# ===== 协同候选特征 (原始构造, 进 GB 前会过 scaler) =====
synergy_candidates = {
    'multi_line_per_finger': MULTILINE / (1.0 + MF3),          # 双指多面比 (BonusTime 高)
    'multiline_x_bpm': MULTILINE * BPM,                         # 多面×BPM
    'density_x_movement': DENSITY * MOVEMENT,                   # 密度×位移
    'crosshand_x_multiline': CROSSHAND * MULTILINE,             # 交叉手×多面
    'movement_per_finger': MOVEMENT / (1.0 + MF3),              # 双指位移比
    'multiline_x_density': MULTILINE * DENSITY,                 # 多面×密度
    'bpm_x_movement': BPM * MOVEMENT,                           # BPM×位移
    'multiline_ratio_norm': MULTILINE / (1.0 + JLINE),          # 多面下落/判定线数
}

gb_names = orig_names + TAIL_NAMES
X_base0 = np.hstack([np.array([[f.get(nn, 0) for nn in orig_names] for f in feats_list]), X_tail])
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

def run_cv(splits, X_base, gn, tag=''):
    oof = np.zeros(n)
    for fi, (tr, te) in enumerate(splits):
        p95_vals, p99_vals = {}, {}
        for j, name in enumerate(gn):
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
    for lv in LV_ORDER:
        m = np.where(levels_arr == lv)[0]
        print(f'  {lv}={mean_absolute_error(y[m], oof[m]):.4f}', end='')
    print()
    return oof

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_base0, y, groups))

# 基线
oof0 = run_cv(splits, X_base0, gb_names, tag='[基线] ')
resid = y - oof0

# 残差与协同候选的相关性 (只看 IN/AT, 排除低段噪声)
print('\n===== 残差(真实-预测) 与 协同候选 的皮尔逊相关 (IN/AT) =====')
mask = (levels_arr == 'IN') | (levels_arr == 'AT')
for name, col in synergy_candidates.items():
    c = col[mask]
    r = y[mask] - oof0[mask]
    if np.std(c) > 0:
        corr = np.corrcoef(c, r)[0, 1]
        print(f'  {name:<26} 与残差相关={corr:+.3f}  (负=特征越高越被低估)')

# 逐个协同特征进 GB 测试
print('\n===== 协同特征进 GB 测试 =====')
for name, col in synergy_candidates.items():
    col2 = col.reshape(-1, 1)
    X_new = np.hstack([X_base0, col2])
    gn2 = gb_names + [name]
    run_cv(splits, X_new, gn2, tag=f'[+{name}] ')

# 组合: 双指多面比 + 多面×BPM (针对 BonusTime)
col_combo = np.column_stack([synergy_candidates['multi_line_per_finger'],
                             synergy_candidates['multiline_x_bpm'],
                             synergy_candidates['movement_per_finger']])
X_combo = np.hstack([X_base0, col_combo])
gn_combo = gb_names + ['multi_line_per_finger', 'multiline_x_bpm', 'movement_per_finger']
run_cv(splits, X_combo, gn_combo, tag='[+3协同特征] ')
