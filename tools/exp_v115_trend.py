# -*- coding: utf-8 -*-
"""实验3: 段内趋势分析 — 官谱各段内 Spearman + level特征诊断
问题: level onehot 占73%重要性 → IN段内排序是否被密度主导/信息不足?
"""
import os, sys, pickle, numpy as np, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = [r for r in cache['official']]
m4 = pickle.load(open(os.path.join(_ROOT, 'models', '6dim_model_v11_4.pkl'), 'rb'))
gb, scaler = m4['gb'], m4['scaler']
FN = m4['feature_names']; LV_ORDER = m4['lv_order']

feats_list = [r['feats'] for r in official]
labels = np.array([r['diff'] for r in official])
levels_list = [r['level'] for r in official]
names_list = [r['name'] for r in official]
n = len(feats_list)
y = labels

# --- 用v11.4模型做OOF预测（不用重新训练：手动5折复现太贵，直接用模型全量预测衡量段内排序）
X_base = np.array([[f.get(nn, 0) for nn in FN] for f in feats_list])
X_lv = np.zeros((n, len(LV_ORDER)))
for i, lv in enumerate(levels_list):
    key = 'IN_AT' if lv in ('IN', 'AT') else lv
    if key in LV_ORDER: X_lv[i, LV_ORDER.index(key)] = 1.0
pred_full = gb.predict(scaler.transform(np.hstack([X_base, X_lv])))
# boost 近似: 用模型 p95 计算 (同app.py, 无条件)
FLAT = m4['MANUAL_FLAT']; CAPS = m4['caps']; P95 = m4['p95_vals']; P99 = m4['p99_vals']
def boost(feats):
    total = 0.0
    cap = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0); pv = P95.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t: continue
        e = v / t - 1.0
        c = CAPS.get(fname, cap)
        if c is not None and e > c: e = c
        total += co * (e ** 0.70)
        if v > max(P99.get(fname, 0), bl * 0.5):
            pe = v / max(P99.get(fname, 0), bl * 0.5) - 1.0
            if c is not None and pe > c: pe = c
            total += co * max(0, pe) ** 0.70 * 0.5
    return total
boosts = np.array([boost(f) for f in feats_list])
pred = pred_full + boosts

errs = pred - y
print('===== 全量模型 (in-sample) 段内排序 =====')
for lo, hi, tag in [(11, 14, '11-14'), (14, 15, '14-15'), (15, 16, '15-16'), (16, 17, '16-17'), (17, 99, '17+')]:
    mk = np.where((y >= lo) & (y < hi))[0]
    if len(mk) < 5: continue
    rho, p = spearmanr(y[mk], pred[mk])
    print(f'  [{tag}]: n={len(mk)} 段内Spearman={rho:.3f} bias={errs[mk].mean():+.3f}')

# IN/AT 单独段内 (无 level 区分度)
for lv, lo, hi in [('IN', 14, 17), ('AT', 15, 99), ('IN', 15, 16.5)]:
    mk = np.where((levels_list == lv) & (y >= lo) & (y < hi))[0]
    if len(mk) < 5: continue
    rho, p = spearmanr(y[mk], pred[mk])
    print(f'  {lv}[{lo}-{hi}]: n={len(mk)} 段内Spearman={rho:.3f} MAE={mean_absolute_error(y[mk], pred[mk]):.3f}')

# ---- 去掉level特征后 段内预测 (诊断level依赖) ----
print('\n===== 无level特征 (仅GB特征) =====')
X_nolv = X_base
gb2 = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3, learning_rate=0.05, subsample=0.8, random_state=42)
sc2 = StandardScaler().fit(X_nolv)
gb2.fit(sc2.transform(X_nolv), y - boosts)
pred2 = gb2.predict(sc2.transform(X_nolv)) + boosts
errs2 = pred2 - y
for lo, hi, tag in [(11, 14, '11-14'), (14, 15, '14-15'), (15, 16, '15-16'), (16, 17, '16-17'), (17, 99, '17+')]:
    mk = np.where((y >= lo) & (y < hi))[0]
    if len(mk) < 5: continue
    rho, p = spearmanr(y[mk], pred2[mk])
    print(f'  [{tag}]: n={len(mk)} 段内Spearman={rho:.3f} bias={errs2[mk].mean():+.3f} MAE={mean_absolute_error(y[mk], pred2[mk]):.3f}')

# ---- level特征重要性 vs 特征值区分度 ----
print('\n===== IN段: level=IN_AT 单类下 GB 特征重要性 top20 =====')
imp = gb.feature_importances_
fn_all = FN + LV_ORDER
pairs = sorted(zip(fn_all, imp), key=lambda x: -x[1])[:25]
for f, i in pairs: print(f'  {i*100:5.2f}%  {f}')

# 段内最佳单特征相关 (IN 14-17)
mk = np.where((levels_list == 'IN') & (y >= 14) & (y <= 17))[0]
print(f'\nIN[14-17] n={len(mk)} 与定数相关性 top10 (去level):')
Xm = X_base[mk]; ym = y[mk]
rs = []
for j, fn in enumerate(FN):
    r = np.corrcoef(Xm[:, j], ym)[0, 1]
    rs.append((abs(r), r, fn))
for a, r, fn in sorted(rs, reverse=True)[:10]:
    print(f'  r={r:+.3f}  {fn}')
print('DONE')
