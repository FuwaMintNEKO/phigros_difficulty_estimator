# -*- coding: utf-8 -*-
"""实验2: 特征精简 — 共线合并/弱相关删除 对官谱CV的影响
用法: python tools/exp_v115_prune.py --mode collapse|both|baseline --out models/6dim_model_v115_X.pkl
"""
import os, sys, pickle, numpy as np, argparse, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='collapse', choices=['baseline', 'collapse', 'both'])
parser.add_argument('--out', default='6dim_model_v115_x.pkl')
parser.add_argument('--rho', type=float, default=0.95)
args = parser.parse_args()

# ---------- 加载官谱特征 ----------
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = [r for r in cache['official']]
print(f'官谱特征: {len(official)}')

# 模型特征名单（与 v11.4 相同）
m4 = pickle.load(open(os.path.join(_ROOT, 'models', '6dim_model_v11_4.pkl'), 'rb'))
gb_feature_names = list(m4['feature_names'])
imp = m4['gb'].feature_importances_
order = np.argsort(-imp)
print(f'GB特征数: {len(gb_feature_names)} (基线)')

feats_list = [r['feats'] for r in official]
labels = np.array([r['diff'] for r in official])
levels_list = [r['level'] for r in official]
names_list = [r['name'] for r in official]
n = len(feats_list)

def build_X(fnames):
    return np.array([[f.get(nn, 0) for nn in fnames] for f in feats_list])

X_full = build_X(gb_feature_names)
y = labels

# ---------- 贪心去冗余 ----------
def collapse_features(fnames, X, rho):
    """按GB重要性降序, 与已保留特征|r|>rho的丢弃"""
    kept, dropped = [], []
    # 按重要性降序
    idx = np.argsort(-np.array([imp[gb_feature_names.index(f)] for f in fnames]))
    cands = [fnames[i] for i in idx]
    # 需要原始顺序的X列索引
    col_of = {f: i for i, f in enumerate(fnames)}
    for f in cands:
        if len(kept) == 0:
            kept.append(f); continue
        # 与所有kept算最大|r|
        xf = X[:, col_of[f]]
        rmax = 0.0
        for g in kept:
            xg = X[:, col_of[g]]
            sd_f, sd_g = xf.std(), xg.std()
            if sd_f < 1e-9 or sd_g < 1e-9:
                r = 1.0 if sd_f == sd_g == 0 else 0.0
            else:
                r = abs(np.corrcoef(xf, xg)[0, 1])
            rmax = max(rmax, r)
            if rmax > rho: break
        if rmax > rho: dropped.append((f, g, round(rmax, 3)))
        else: kept.append(f)
    return kept, dropped

fnames_sel = list(gb_feature_names)
if args.mode in ('collapse', 'both'):
    kept, dropped = collapse_features(fnames_sel, X_full, args.rho)
    print(f'共线去冗余: {len(fnames_sel)} -> {len(kept)} (丢弃{len(dropped)})')
    fnames_sel = kept
if args.mode == 'both':
    # 弱相关删除: |r(y)|<0.05 且 非level/核心密度
    Xs = build_X(fnames_sel)
    weak = []
    for i, f in enumerate(fnames_sel):
        r = np.corrcoef(Xs[:, i], y)[0, 1]
        if abs(r) < 0.05:
            weak.append((f, round(r, 3)))
    keep2 = [f for f in fnames_sel if f not in [w[0] for w in weak]]
    print(f'弱相关删除: {len(fnames_sel)} -> {len(keep2)}')
    for w in weak: print(f'    drop {w[0]} r={w[1]}')
    fnames_sel = keep2
print(f'最终GB特征数: {len(fnames_sel)}')

# ---------- CV (与 train_v11_a 完全一致) ----------
X_base = build_X(fnames_sel)
LV_ORDER = ['EZ', 'HD', 'IN_AT']
X_lv = np.zeros((n, len(LV_ORDER)))
for i, lv in enumerate(levels_list):
    key = 'IN_AT' if lv in ('IN', 'AT') else lv
    X_lv[i, LV_ORDER.index(key)] = 1.0
levels_arr = np.array(levels_list)
SAMPLE_W = np.ones(n)
SAMPLE_W[levels_arr == 'EZ'] = 1.5
SAMPLE_W[levels_arr == 'HD'] = 1.5
hi_mask = (levels_arr == 'AT') | ((levels_arr == 'IN') & (y >= 16.0))
SAMPLE_W[hi_mask] = 1.0

FLAT = m4['MANUAL_FLAT']; CAPS = m4['caps']
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

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_base, y, groups=np.array(names_list)))
oof = np.zeros(n)
for fi, (tr, te) in enumerate(splits):
    p95_vals, p99_vals = {}, {}
    for j, name in enumerate(fnames_sel):
        col = X_base[tr, j]
        p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
        p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
    boosts = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in feats_list])
    X_tr = np.hstack([X_base[tr], X_lv[tr]])
    X_te = np.hstack([X_base[te], X_lv[te]])
    sc = StandardScaler().fit(X_tr)
    gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
    gb.fit(sc.transform(X_tr), y[tr] - boosts[tr], sample_weight=SAMPLE_W[tr])
    oof[te] = gb.predict(sc.transform(X_te)) + boosts[te]
    print(f'  fold{fi} OOF完成', flush=True)

errs = oof - y
print(f'\n===== 官谱歌曲分组CV [{args.mode}] =====')
print(f'整体MAE = {mean_absolute_error(y, oof):.4f} | bias = {errs.mean():+.4f}')
for lv in ['EZ', 'HD', 'IN', 'AT']:
    mk = np.where(levels_arr == lv)[0]
    if len(mk): print(f'  {lv}: n={len(mk)} MAE={mean_absolute_error(y[mk], oof[mk]):.4f} bias={errs[mk].mean():+.4f}')
for lo, hi, tag in [(11, 14, '11-14'), (14, 15, '14-15'), (15, 16, '15-16'), (16, 17, '16-17'), (17, 99, '17+')]:
    mk = np.where((y >= lo) & (y < hi))[0]
    if len(mk): print(f'  定数[{tag}]: n={len(mk)} MAE={mean_absolute_error(y[mk], oof[mk]):.4f} bias={errs[mk].mean():+.4f}')
# 多指/双指
for lo in [16.0]:
    mk = np.where(y >= lo)[0]
    if len(mk) == 0: continue
    g_mf = [i for i in mk if feats_list[i].get('multi_finger_3plus_events', 0) >= 30]
    g_df = [i for i in mk if feats_list[i].get('multi_finger_3plus_events', 0) <= 5]
    for g, tag in [(g_mf, '多指'), (g_df, '双指')]:
        if len(g): print(f'  >= {lo} [{tag}]: n={len(g)} bias={errs[g].mean():+.4f} MAE={mean_absolute_error(y[g], oof[g]):.4f}')
print('DONE-CV')
