# -*- coding: utf-8 -*-
"""双向互证: 纯自制谱(社区定数)训练模型 → 预测官谱, 掌握两个定数标尺的差距。
方向A(已做): 官谱模型→预测自制谱 MAE 0.537
方向B(本脚本): 自制谱模型→预测官谱, 看社区定数标尺 vs 官方定数标尺的系统性偏移
"""
import os, sys, io, json
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator'
sys.path.insert(0, ROOT)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

CACHE = os.path.join(ROOT, 'data', 'phira', '_feats_cache_custom.npz')
CHARTS = os.path.join(ROOT, 'data', 'phira', 'charts.json')
JSON_DIR = os.path.join(ROOT, 'data', 'phira', 'json')

def parse_level(lv_str):
    s = str(lv_str).strip().upper()
    for lv in ['AT', 'IN', 'HD', 'EZ']:
        if s.startswith(lv):
            return lv
    return None

# ===== 1. 提取自制谱特征 (筛选: 上架 + 11<=diff<=17.6 + IN/AT) =====
if os.path.exists(CACHE):
    print('加载自制谱特征缓存...')
    d = np.load(CACHE, allow_pickle=True)
    cf_feats, cf_labels, cf_levels, cf_names, cf_ids = (d['feats'], d['labels'], d['levels'],
                                                         d['names'], d['ids'])
else:
    charts = json.load(open(CHARTS, encoding='utf-8'))
    ranked = charts.get('上架', [])
    meta = {c['id']: c for c in ranked}
    feats_list, labels, levels, names, ids = [], [], [], [], []
    for c in ranked:
        diff = c.get('difficulty', 0)
        if not (11 <= diff <= 17.6):
            continue
        lv = parse_level(c.get('level', ''))
        if lv not in ('IN', 'AT'):
            continue
        cid = c['id']
        path = os.path.join(JSON_DIR, f'{cid}.json')
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'rb') as f:
                raw = f.read()
            cd, _ = load_chart_from_bytes(raw)
            if cd is None:
                continue
            feats = extract_features(cd, speed=1.0)
            if not feats:
                continue
            feats_list.append(feats); labels.append(diff)
            levels.append(lv); names.append(c.get('name', '')); ids.append(cid)
        except Exception:
            pass
    cf_feats = np.array(feats_list, dtype=object)
    cf_labels = np.array(labels); cf_levels = np.array(levels)
    cf_names = np.array(names); cf_ids = np.array(ids)
    np.savez(CACHE, feats=cf_feats, labels=cf_labels, levels=cf_levels,
             names=cf_names, ids=cf_ids)
    print(f'自制谱特征缓存已保存: {CACHE}')

n_c = len(cf_feats)
print(f'自制谱训练集: {n_c} 张 (IN/AT, 定数11-17.6)')
print(f'  IN={sum(cf_levels=="IN")}  AT={sum(cf_levels=="AT")}')
print(f'  定数范围 {cf_labels.min():.1f}~{cf_labels.max():.1f}, 均值 {cf_labels.mean():.2f}')

# ===== 2. 加载官谱特征 (复用缓存) =====
od = np.load(os.path.join(ROOT, 'data', 'phira', '_feats_cache.npz'), allow_pickle=True)
of_feats = od['feats_list']; of_labels = od['labels']
of_levels = od['levels_list']; of_names = od['names_list']
gb_feature_names = list(od['gb_feature_names'])
n_o = len(of_feats)
print(f'\n官谱: {n_o} 张, GB特征 {len(gb_feature_names)}')

# ===== 3. 构造特征矩阵 (双方都用 gb_feature_names + IN/AT onehot) =====
def build_matrix(feats_list, levels):
    X = np.array([[f.get(nn, 0) for nn in gb_feature_names] for f in feats_list])
    X_lv = np.zeros((len(feats_list), 2))
    X_lv[:, 0] = (levels == 'IN').astype(float)
    X_lv[:, 1] = (levels == 'AT').astype(float)
    return np.hstack([X, X_lv])

X_c = build_matrix(cf_feats, cf_levels)
y_c = cf_labels.astype(float)
X_o = build_matrix(of_feats, of_levels)
y_o = of_labels.astype(float)

# ===== 4. 训练自制谱模型 (纯GB, 无boost) =====
print('\n训练自制谱模型 (GB 500树)...')
sc = StandardScaler().fit(X_c)
gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                               learning_rate=0.05, subsample=0.8, random_state=42)
gb.fit(sc.transform(X_c), y_c)
# 自评估 (in-sample)
pred_c = gb.predict(sc.transform(X_c))
print(f'自制谱 in-sample MAE = {mean_absolute_error(y_c, pred_c):.4f}')

# ===== 5. 预测官谱 (IN/AT 段) =====
mask_o = (of_levels == 'IN') | (of_levels == 'AT')
pred_o = gb.predict(sc.transform(X_o[mask_o]))
y_o_ia = y_o[mask_o]
lv_o_ia = of_levels[mask_o]
print(f'\n===== 自制谱模型 预测官谱 (IN/AT段, n={mask_o.sum()}) =====')
print(f'整体MAE = {mean_absolute_error(y_o_ia, pred_o):.4f}')
print(f'整体偏差(预测-真实) = {(pred_o - y_o_ia).mean():+.4f}')

for lo, hi, label in [(14, 15, '14-15'), (15, 16, '15-16'), (16, 16.5, '16-16.5'),
                       (16.5, 17, '16.5-17'), (17, 17.6, '17-17.6')]:
    m = (y_o_ia >= lo) & (y_o_ia < hi)
    if m.sum() == 0:
        continue
    b = pred_o[m] - y_o_ia[m]
    print(f'  官谱定数[{label}]: n={m.sum():>3} 均值偏差={b.mean():+.3f} MAE={mean_absolute_error(y_o_ia[m], pred_o[m]):.3f}')

for lv in ['IN', 'AT']:
    m = lv_o_ia == lv
    b = pred_o[m] - y_o_ia[m]
    print(f'  官谱 {lv}: n={m.sum():>3} 均值偏差={b.mean():+.3f} MAE={mean_absolute_error(y_o_ia[m], pred_o[m]):.3f}')

# 最被高估/低估的官谱 (自制谱模型视角)
print('\n===== 自制谱模型 最"高估"的官谱 (预测 >> 真实) =====')
order = np.argsort(-(pred_o - y_o_ia))
for i in order[:10]:
    print(f'  {of_names[mask_o][i][:30]:<30} {lv_o_ia[i]} 真实={y_o_ia[i]:.1f} 自制模型预测={pred_o[i]:.2f} 差={pred_o[i]-y_o_ia[i]:+.2f}')
print('===== 自制谱模型 最"低估"的官谱 (预测 << 真实) =====')
for i in order[-10:][::-1]:
    print(f'  {of_names[mask_o][i][:30]:<30} {lv_o_ia[i]} 真实={y_o_ia[i]:.1f} 自制模型预测={pred_o[i]:.2f} 差={pred_o[i]-y_o_ia[i]:+.2f}')
