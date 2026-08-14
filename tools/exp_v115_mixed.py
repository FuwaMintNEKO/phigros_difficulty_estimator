# -*- coding: utf-8 -*-
"""混合训练实验: 官谱982 + 上架589(社区定数) 训练, 评估官谱CV + 上架谱效果
用法: python tools/exp_v115_mixed.py --w 0.5 --out models/6dim_model_v115m_w05.pkl
"""
import os, sys, pickle, numpy as np, io, argparse, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

parser = argparse.ArgumentParser()
parser.add_argument('--w', type=float, default=0.5, help='社区谱样本权重')
parser.add_argument('--out', default='6dim_model_v115m.pkl')
parser.add_argument('--ranked-rc', type=int, default=0, help='上架谱最低ratingCount(0=全用)')
parser.add_argument('--subset', action='store_true', help='只用极端子集(未上架高评分多指低密被低估)')
parser.add_argument('--sub-w', type=float, default=1.0, help='子集权重')
args = parser.parse_args()

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = cache['official']
ranked = [r for r in cache['ranked'] if r['diff'] and 10 < r['diff'] < 25]
print(f'官谱: {len(official)} 上架谱: {len(ranked)}')

def build(feats_list, levels_list):
    n = len(feats_list)
    feats_all = []
    for r in feats_list:
        feats_all.append(r['feats'])
    X = np.array([[f.get(nn, 0) for nn in gb_fn] for f in feats_all])
    return X

# 官谱特征名(与v11.5c一致): 从模型取
m_ref = pickle.load(open(os.path.join(_ROOT, 'models', '6dim_model_v11_5c.pkl'), 'rb'))
gb_fn = list(m_ref['feature_names'])
LV_ORDER = ['EZ', 'HD', 'IN_AT']

def onehot(level):
    key = 'IN_AT' if level in ('IN', 'AT') else level
    vec = [0.0]*3
    if key in LV_ORDER: vec[LV_ORDER.index(key)] = 1.0
    else: vec[2] = 1.0
    return vec

# 官谱
Xo = np.array([[f['feats'].get(nn, 0) for nn in gb_fn] for f in official])
Xo_lv = np.array([onehot(f['level']) for f in official])
yo = np.array([f['diff'] for f in official])
names_o = np.array([f['name'] for f in official])
lv_o = np.array([f['level'] for f in official])
# 上架 / 子集
if args.subset:
    sub = pickle.load(open(os.path.join(_ROOT, 'data', 'phira', 'extreme_subset.pkl'), 'rb'))['charts']
    ranked = sub
    print(f'极端子集: {len(sub)} 张')
Xr = np.array([[f['feats'].get(nn, 0) for nn in gb_fn] for f in ranked])
Xr_lv = np.array([onehot(f['level'].upper() if f['level'] else 'IN') for f in ranked])
yr = np.array([f['diff'] for f in ranked])
print(f'官谱 {Xo.shape} 上架/子集 {Xr.shape}')

# 样本权重
W = np.ones(len(official))
W[lv_o == 'EZ'] = 1.5
W[lv_o == 'HD'] = 1.5
Wr = np.full(len(ranked), args.sub_w if args.subset else args.w)

def compute_boost(feats, p95_vals, p99_vals):
    total = 0.0
    cap = 4.0
    for fname, bl, co in MANUAL_FLAT:
        v = feats.get(fname, 0)
        pv = p95_vals.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t: continue
        e = v / t - 1.0
        if e > cap: e = cap
        x = co * (e ** 0.70)
        if v > max(p99_vals.get(fname, 0), bl * 0.5):
            pe = v / max(p99_vals.get(fname, 0), bl * 0.5) - 1.0
            if pe > cap: pe = cap
            x += co * max(0, pe) ** 0.70 * 0.5
        total += x
    return total

# 官谱5折CV评估 (只在官谱上)
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(Xo, yo, groups=names_o))
oof = np.zeros(len(official))
# 每折: p95/p99 用该折训练官谱
for fi, (tr, te) in enumerate(splits):
    p95_vals, p99_vals = {}, {}
    for j, name in enumerate(gb_fn):
        col = Xo[tr, j]
        p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
        p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
    boosts_o = np.array([compute_boost(f['feats'], p95_vals, p99_vals) for f in official])
    # 混合训练: 该折官谱训练 + 全部上架
    X_tr = np.vstack([np.hstack([Xo[tr], Xo_lv[tr]]), np.hstack([Xr, Xr_lv])])
    y_tr = np.concatenate([yo[tr] - boosts_o[tr], yr])
    w_tr = np.concatenate([W[tr], Wr])
    sc = StandardScaler().fit(X_tr)
    gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
    gb.fit(sc.transform(X_tr), y_tr, sample_weight=w_tr)
    oof[te] = gb.predict(sc.transform(np.hstack([Xo[te], Xo_lv[te]]))) + boosts_o[te]
    print(f'  fold{fi} 完成 (混合n={len(X_tr)})', flush=True)

errs = oof - yo
print(f'\n===== 官谱CV [混合 w={args.w}] =====')
print(f'MAE={mean_absolute_error(yo, oof):.4f} bias={errs.mean():+.4f}')
for lv in ['EZ', 'HD', 'IN', 'AT']:
    mk = np.where(lv_o == lv)[0]
    if len(mk): print(f'  {lv}: n={len(mk)} MAE={mean_absolute_error(yo[mk], oof[mk]):.4f} bias={errs[mk].mean():+.4f}')

# 保存混合模型 (全量训练)
p95_vals, p99_vals = {}, {}
for j, name in enumerate(gb_fn):
    col = Xo[:, j]
    p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
    p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
boosts_o = np.array([compute_boost(f['feats'], p95_vals, p99_vals) for f in official])
X_all = np.vstack([np.hstack([Xo, Xo_lv]), np.hstack([Xr, Xr_lv])])
y_all = np.concatenate([yo - boosts_o, yr])
w_all = np.concatenate([W, Wr])
sc = StandardScaler().fit(X_all)
gb_f = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                 learning_rate=0.05, subsample=0.8, random_state=42)
gb_f.fit(sc.transform(X_all), y_all, sample_weight=w_all)
model = {'gb': gb_f, 'scaler': sc, 'feature_names': gb_fn,
         'p95_vals': p95_vals, 'p99_vals': p99_vals, 'lv_order': LV_ORDER,
         'version': f'v115m-w{args.w}-sub{args.sub_w if args.subset else 0}', 'n_train': len(X_all),
         'MANUAL_FLAT': MANUAL_FLAT, 'caps': {'_default': 4.0},
         'train_meta': {'n': len(X_all), 'songs': len(set(names_o)),
                        'cv_mae': float(mean_absolute_error(yo, oof)),
                        'w': args.w, 'n_ranked': len(ranked), 'subset': args.subset}}
path = os.path.join(_ROOT, 'models', args.out)
with open(path, 'wb') as f:
    pickle.dump(model, f)
print(f'已保存: {path}')
print('DONE')
