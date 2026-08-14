"""
v8.3: density_dimension = √(rcnps × above_avg_mean) 重训
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os, pickle, numpy as np, math, re
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, '.')
from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from unified_parser import load_chart_from_bytes
import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print('='*70); print('  v8.3 — density=√(rcnps×above_avg_mean)'); print('='*70)

with open('models/6dim_model_v8_2.pkl', 'rb') as f:
    v82 = pickle.load(f)
P95o = v82['p95_vals']; P99o = v82['p99_vals']
DC = v82['dynamic_cap']; FLATo = v82['FLAT_FEATURES']
FNo = v82['feature_names']

# 替换 density_dimension 的 P95
CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)

all_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    for lv in ['IN','AT']:
        if lv in info['levels'] and lv in diffs:
            all_items.append({'folder':fn,'filepath':info['levels'][lv],'difficulty':diffs[lv],'level':lv})

feats_list, labels = [], []
for item in all_items:
    try:
        cd = load_chart_json(item['filepath'])
        feats = extract_features(cd)
        if feats: feats_list.append(feats); labels.append(item['difficulty'])
    except: pass

# 更新density_dimension的P95/P99
vals = [f.get('density_dimension', 0) for f in feats_list]
P95 = dict(P95o); P99 = dict(P99o)
P95['density_dimension'] = float(np.percentile(vals, 95))
P99['density_dimension'] = float(np.percentile(vals, 99))
print(f'density_dimension: P95={P95["density_dimension"]:.2f} P99={P99["density_dimension"]:.2f} (was P95={P95o["density_dimension"]:.2f})')

# Boost: same as v8.2 (no new features)
FLAT = list(FLATo)
feat_names_boost = [f[0] for f in FLAT]

n_all = len(feats_list); labels = np.array(labels)
print(f'IN/AT: {n_all}, 难度 {labels.min():.1f}~{labels.max():.1f}')

def compute_excess(feats, fname, bl):
    val = feats.get(fname, 0)
    pv = P95.get(fname, 0)
    thresh = max(pv * 0.55, bl * 0.5)
    if val <= thresh: return 0.0
    excess = (val / thresh - 1.0) ** 0.70
    if val > max(P99.get(fname, 0), bl * 0.5):
        pe = (val / max(P99.get(fname, 0), bl * 0.5) - 1.0)
        excess += 0.5 * max(0, pe) ** 0.70
    return excess

n_boost = len(FLAT)
X_excess = np.zeros((n_all, n_boost))
for i in range(n_all):
    for j, (fname, bl, _) in enumerate(FLAT):
        X_excess[i, j] = compute_excess(feats_list[i], fname, bl)

FN = list(FNo)
X_gb = np.array([[f.get(n, 0) for n in FN] for f in feats_list])
y = labels.copy()

def _dc(r):
    if r <= DC['knee']: return r
    return DC['knee'] + (r - DC['knee']) ** DC['power']

old_co = {f[0]: f[2] for f in FLAT}
old_bl = {f[0]: f[1] for f in FLAT}
co_current = np.array([old_co[fn] for fn in feat_names_boost])

def compute_raw_boost(feats, co_arr):
    raw = 0.0
    for j, (fname, bl, _) in enumerate(FLAT):
        raw += co_arr[j] * compute_excess(feats, fname, bl)
    return raw

def adjust_boost(boost, gb_val, target=0.28, thresh=0.22, power=0.75):
    if boost < 2.0 or gb_val <= 0: return boost
    r = boost / gb_val; e = target * gb_val
    a = e * ((boost / e) ** power)
    w = 1 / (1 + math.exp(-25 * (r - thresh)))
    return (1 - w) * boost + w * a

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
bins = np.digitize(y, bins=[12,13,14,15,16,17,18])
train_idx, test_idx = next(sss.split(X_gb, bins))

for it in range(3):
    print(f'\n--- 迭代 {it+1}/3 ---')
    all_boosts = np.array([_dc(compute_raw_boost(feats_list[i], co_current)) for i in range(n_all)])
    X_tr, y_tr = X_gb[train_idx], y[train_idx]; X_te = X_gb[test_idx]; y_te = y[test_idx]
    b_tr, b_te = all_boosts[train_idx], all_boosts[test_idx]
    
    sc = StandardScaler()
    gbm = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
    gbm.fit(sc.fit_transform(X_tr), y_tr - b_tr)
    preds = gbm.predict(sc.transform(X_te)) + b_te
    r2 = r2_score(y_te, preds); mae = mean_absolute_error(y_te, preds)
    imp = dict(zip(FN, gbm.feature_importances_))
    print(f'  GB: R²={r2:.4f}, MAE={mae:.4f}  density_dim={imp.get("density_dimension",0):.4f}')
    
    sc_all = StandardScaler()
    gb_full = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                         learning_rate=0.05, subsample=0.8, random_state=42)
    gb_full.fit(sc_all.fit_transform(X_gb), y - all_boosts)
    y_res = y - gb_full.predict(sc_all.transform(X_gb))
    
    best_a = 5.0; best_cv = float('inf')
    for a in [0.01, 0.1, 1, 5, 10, 50, 100]:
        ridge = Ridge(alpha=a, fit_intercept=False, positive=True)
        s = -cross_val_score(ridge, X_excess, y_res, cv=5, scoring='neg_mean_absolute_error')
        if s.mean() < best_cv: best_cv = s.mean(); best_a = a
    ridge = Ridge(alpha=best_a, fit_intercept=False, positive=True)
    ridge.fit(X_excess, y_res); co_new = ridge.coef_
    
    for k in ['density_dimension','fast_note_density_16th','tap_per_second','total_notes']:
        if k in feat_names_boost:
            print(f'  {k:<35} co={co_new[feat_names_boost.index(k)]:.4f}')
    co_current = 0.3 * co_current + 0.7 * co_new

print('\n--- 全量训练 ---')
all_boosts_f = np.array([_dc(compute_raw_boost(feats_list[i], co_current)) for i in range(n_all)])
sc_f = StandardScaler()
gb_f = GradientBoostingRegressor(n_estimators=700, max_depth=5, min_samples_leaf=3,
                                  learning_rate=0.05, subsample=0.8, random_state=42)
gb_f.fit(sc_f.fit_transform(X_gb), y - all_boosts_f)

# 测试
test_dir = r'C:\Users\NaNK\Downloads'
chart_data = []
for fn in os.listdir(test_dir):
    if not fn.endswith('.json'): continue
    if '_2xBPM' in fn or '_2x' in fn: continue
    fp = os.path.join(test_dir, fn)
    if os.path.getsize(fp) < 100: continue
    try:
        rating = None
        for m in re.finditer(r'\((\d+\.?\d*)\)', fn):
            val = float(m.group(1))
            if 5 <= val <= 20: rating = val; break
        if rating is None: rating = 0
        with open(fp, 'rb') as f: raw = f.read()
        data, _ = load_chart_from_bytes(raw)
        feats = extract_features(data)
        if feats: chart_data.append((fn, feats, rating))
    except: continue

best = None
for target in [0.26, 0.28, 0.30]:
    for power in [0.65, 0.70, 0.75, 0.80]:
        for thresh in [0.20, 0.22, 0.24, 0.26]:
            for cf in [1.00, 1.02, 1.04, 1.06]:
                errs = []
                for fn, feats, rating in chart_data:
                    X = np.array([[feats.get(k,0) for k in FN]])
                    pg = float(gb_f.predict(sc_f.transform(X))[0])
                    raw = compute_raw_boost(feats, co_current * cf)
                    pb = _dc(raw)
                    pa = adjust_boost(pb, pg, target=target, thresh=thresh, power=power)
                    errs.append(pg + pa - rating)
                rated = [e for e, (_,_,r) in zip(errs, chart_data) if r > 0]
                if not rated: continue
                m = np.mean([abs(e) for e in rated])
                pos = sum(1 for e, (_,_,r) in zip(errs, chart_data) if r > 0 and e > 0.01)
                neg = sum(1 for e, (_,_,r) in zip(errs, chart_data) if r > 0 and e < -0.01)
                if abs(pos-neg) <= 6 and m < 0.50:
                    if best is None or m < best[5]:
                        best = (target, power, thresh, cf, abs(pos-neg), m, pos, neg)

if best:
    t, p, th, cf, bal, mae, pos, neg = best
    print(f'\n最优: t={t} p={p} th={th} cf={cf} MAE={mae:.3f} +{pos}/-{neg}')
    
    with open('models/6dim_model_v8_2.pkl', 'rb') as f:
        v82m = pickle.load(f)
    gb_o = v82m['gb']; sc_o = v82m['scaler']; fn_o = v82m['feature_names']
    flat_o = v82m['FLAT_FEATURES']; dc_o = v82m['dynamic_cap']
    
    key_tags = ['Final EndGame','朧月','Apollo','恋ひ恋ふ縁','怪文書','Waking','ex7','ex8','Regrets','silly','Submerged','Cheerio']
    
    print(f'\n{"谱面":<14} {"定数":>5} {"v8.2":>7} {"v8.3":>7} {"Δ":>7} {"ddim":>6} {"above":>6} {"above%":>6}')
    print('-' * 68)
    
    for fn, feats, rating in chart_data:
        matched = False
        for tag in key_tags:
            if tag in fn: matched = True; break
        if not matched: continue
        
        Xo = np.array([[feats.get(k,0) for k in fn_o]])
        po_gb = float(gb_o.predict(sc_o.transform(Xo))[0])
        po_raw = 0.0
        for fname, bl, co in flat_o:
            v = feats.get(fname, 0); pv = P95.get(fname, 0)
            th = max(pv * 0.55, bl * 0.5)
            if v <= th: continue
            e = (v/th - 1.0) ** 0.70
            if v > max(P99.get(fname,0), bl * 0.5):
                pe = (v / max(P99.get(fname,0), bl * 0.5) - 1.0)
                e += 0.5 * max(0, pe) ** 0.70
            po_raw += co * e
        po_b = po_raw if po_raw <= dc_o['knee'] else dc_o['knee'] + (po_raw - dc_o['knee']) ** dc_o['power']
        po_a = adjust_boost(po_b, po_gb, target=0.28, thresh=0.22, power=0.75)
        po_pred = po_gb + po_a
        
        Xn = np.array([[feats.get(k,0) for k in FN]])
        pn_gb = float(gb_f.predict(sc_f.transform(Xn))[0])
        pn_raw = compute_raw_boost(feats, co_current * cf)
        pn_b = _dc(pn_raw)
        pn_a = adjust_boost(pn_b, pn_gb, target=t, thresh=th, power=p)
        pn_pred = pn_gb + pn_a
        
        short = fn[:14]
        am = feats.get('above_avg_density_mean', 0)
        ar = feats.get('above_avg_density_ratio', 0)
        print(f'{short:<14} {rating:>5.1f} {po_pred:>7.2f} {pn_pred:>7.2f} {pn_pred-po_pred:>+7.2f} {feats.get("density_dimension",0):>6.1f} {am:>6.1f} {ar:>6.0%}')

# 保存
scaled_co = co_current * cf
FLAT_F = [(fname, old_bl.get(fname, 1.0), float(scaled_co[j])) for j, fname in enumerate(feat_names_boost)]
out = {'gb': gb_f, 'scaler': sc_f, 'feature_names': list(FN),
       'p95_vals': P95, 'p99_vals': P99, 'FLAT_FEATURES': FLAT_F, 'dynamic_cap': DC,
       'sigmoid_params': {'target': t, 'power': p, 'thresh': th}}
os.makedirs('models', exist_ok=True)
with open('models/6dim_model_v8_3.pkl', 'wb') as f: pickle.dump(out, f)
print(f'\n已保存: models/6dim_model_v8_3.pkl')
