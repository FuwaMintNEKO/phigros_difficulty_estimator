# -*- coding: utf-8 -*-
"""实验: kyou标签(6类)one-hot 加入训练 — CV对比
用法: python tools/exp_v1110_kyou_feat.py --mode base|kyou
"""
import os, sys, pickle, numpy as np, io, argparse, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='kyou', choices=['base', 'kyou'])
parser.add_argument('--out', default='6dim_model_v1110_kyou.pkl')
args = parser.parse_args()

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = cache['official']
m_ref = pickle.load(open(os.path.join(_ROOT, 'models', '6dim_model_v11_9.pkl'), 'rb'))
gb_fn = list(m_ref['feature_names'])
LV_ORDER = ['EZ', 'HD', 'IN_AT']
# kyou 标签
kyou = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'kyou_tags.json'), encoding='utf-8'))
KTAGS = ['硬抗', '综合', '定位', '读谱', '拆谱', '多指']
def norm(s):
    return re.sub(r'[^a-z0-9\u4e00-\u9fff]', '', s.lower())
kyou_map = [(norm(k['song']), k['tag'].replace('?', '').strip()) for k in kyou]

def kyou_vec(name):
    kn = norm(name)
    v = [0.0]*len(KTAGS)
    for kkn, ktag in kyou_map:
        if len(kkn) >= 4 and (kkn in kn or kn in kkn):
            if ktag in KTAGS: v[KTAGS.index(ktag)] = 1.0
            break
    return v + [1.0 if any(v) else 0.0]  # + has_tag

def onehot(level):
    key = 'IN_AT' if level in ('IN', 'AT') else level
    vec = [0.0]*3
    if key in LV_ORDER: vec[LV_ORDER.index(key)] = 1.0
    else: vec[2] = 1.0
    return vec

n = len(official)
X_base = np.array([[r['feats'].get(nn, 0) for nn in gb_fn] for r in official])
X_lv = np.array([onehot(r['level']) for r in official])
if args.mode == 'kyou':
    X_kyou = np.array([kyou_vec(r['name']) for r in official])
    X_all = np.hstack([X_base, X_lv, X_kyou])
    print(f'kyou特征: 命中 {int(np.sum(X_kyou[:,-1]))}/{n}')
else:
    X_all = np.hstack([X_base, X_lv])
y = np.array([r['diff'] for r in official])
names_o = np.array([r['name'] for r in official])
lv_o = np.array([r['level'] for r in official])
SAMPLE_W = np.ones(n)
SAMPLE_W[lv_o == 'EZ'] = 1.5
SAMPLE_W[lv_o == 'HD'] = 1.5

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

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_all, y, groups=names_o))
oof = np.zeros(n)
for fi, (tr, te) in enumerate(splits):
    p95_vals, p99_vals = {}, {}
    for j, name in enumerate(gb_fn):
        col = X_base[tr, j]
        p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
        p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
    boosts = np.array([compute_boost(r['feats'], p95_vals, p99_vals) for r in official])
    sc = StandardScaler().fit(X_all[tr])
    gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
    gb.fit(sc.transform(X_all[tr]), y[tr] - boosts[tr], sample_weight=SAMPLE_W[tr])
    oof[te] = gb.predict(sc.transform(X_all[te])) + boosts[te]
errs = oof - y
print(f'===== 官谱CV [{args.mode}] =====')
print(f'MAE={mean_absolute_error(y, oof):.4f} bias={errs.mean():+.4f}')
for lv in ['EZ', 'HD', 'IN', 'AT']:
    mk = np.where(lv_o == lv)[0]
    if len(mk): print(f'  {lv}: n={len(mk)} MAE={mean_absolute_error(y[mk], oof[mk]):.4f}')
# 保存 (全量)
p95_vals, p99_vals = {}, {}
for j, name in enumerate(gb_fn):
    col = X_base[:, j]
    p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
    p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
boosts = np.array([compute_boost(r['feats'], p95_vals, p99_vals) for r in official])
sc = StandardScaler().fit(X_all)
gb_f = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                 learning_rate=0.05, subsample=0.8, random_state=42)
gb_f.fit(sc.transform(X_all), y - boosts, sample_weight=SAMPLE_W)
model = {'gb': gb_f, 'scaler': sc, 'feature_names': gb_fn,
         'p95_vals': p95_vals, 'p99_vals': p99_vals, 'lv_order': LV_ORDER,
         'version': f'v1110-{args.mode}', 'n_train': n,
         'MANUAL_FLAT': MANUAL_FLAT, 'caps': {'_default': 4.0},
         'train_meta': {'n': n, 'songs': len(set(names_o)), 'cv_mae': float(mean_absolute_error(y, oof)), 'mode': args.mode},
         'kyou_feats': KTAGS + ['has_tag'] if args.mode == 'kyou' else []}
pickle.dump(model, open(os.path.join(_ROOT, 'models', args.out), 'wb'))
print(f'已保存: {args.out}')
print('DONE')
