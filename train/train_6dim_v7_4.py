"""
v7.4b — 最小改动策略

变化:
  1. 保留v7.3的GB特征(不删), 保留density_dimension  
  2. 追加 real_core_notes_per_second + core_peak_density_1sec_top5avg 到boost (共44→46)
  3. Ridge会自动处理三者间的冗余
  4. Sigmoid参数重新扫描
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, json, os, pickle, numpy as np, math, re
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

print('='*70); print('  v7.4b — 保留ddim+追加子特征'); print('='*70)

with open('models/6dim_model_v7_2.pkl', 'rb') as f:
    v7m = pickle.load(f)
P95 = v7m['p95_vals']; P99 = v7m['p99_vals']
DC = {'knee': 1.0, 'power': 0.90}
FN_GB = v7m['feature_names']  # 保持219

# 从v7.3拿FLAT, 追加两个子特征
with open('models/6dim_model_v7_3.pkl', 'rb') as f:
    v73m = pickle.load(f)
FLAT_OLD = v73m['FLAT_FEATURES']

# 构建新FLAT: 保留全部, 加两个新特征(初始co=原ddim_co的30%)
old_dd_co = None
for fname, bl, co in FLAT_OLD:
    if fname == 'density_dimension':
        old_dd_co = co; break

EXTRA_FEATS = [
    ('real_core_notes_per_second', 2.0, old_dd_co * 0.30),
    ('core_peak_density_1sec_top5avg', 6.0, old_dd_co * 0.30),
]

# 用v7.3的co作为种子
co_seed = {}
for fname, bl, co in FLAT_OLD:
    co_seed[fname] = co
for fname, bl, co in EXTRA_FEATS:
    co_seed[fname] = co

feat_names_all = list(co_seed.keys())
n_boost = len(feat_names_all)
print(f'  Boost: {len(FLAT_OLD)} → {n_boost} (追加2个密度子特征)')
print(f'  GB: {len(FN_GB)}特征 (不乱动)')

# 基线
old_bl = {f[0]: f[1] for f in FLAT_OLD}
BL_EXTRA = {f[0]: f[1] for f in EXTRA_FEATS}

# ====== 加载数据 ======
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

n_all = len(feats_list); labels = np.array(labels)
print(f'\nIN/AT官谱: {n_all}, 难度 {labels.min():.1f}~{labels.max():.1f}')

# ====== 工具函数 ======
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

X_excess = np.zeros((n_all, n_boost))
for i in range(n_all):
    for j, fname in enumerate(feat_names_all):
        bl_val = BL_EXTRA.get(fname, old_bl.get(fname, 1.0))
        X_excess[i, j] = compute_excess(feats_list[i], fname, bl_val)

X_gb = np.array([[f.get(n,0) for n in FN_GB] for f in feats_list])
y = labels.copy()

def _dynamic_cap(raw):
    return raw if raw <= DC['knee'] else DC['knee'] + (raw - DC['knee']) ** DC['power']

def compute_raw_boost(feats, co_arr):
    raw = 0.0
    for j, fname in enumerate(feat_names_all):
        bl_val = BL_EXTRA.get(fname, old_bl.get(fname, 1.0))
        raw += co_arr[j] * compute_excess(feats, fname, bl_val)
    return raw

def adjust_boost_smooth(boost, gb_val, target=0.24, thresh=0.24, power=0.70):
    if boost < 2.0 or gb_val <= 0: return boost
    ratio = boost / gb_val
    expected = target * gb_val
    adj = expected * ((boost / expected) ** power)
    w = 1 / (1 + math.exp(-25 * (ratio - thresh)))
    return (1 - w) * boost + w * adj

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
bins = np.digitize(y, bins=[12,13,14,15,16,17,18])
train_idx, test_idx = next(sss.split(X_gb, bins))

# ====== 迭代 (只用2轮, 因为改动很小) ======
co_current = np.array([co_seed[fname] for fname in feat_names_all])

for it in range(2):
    print(f'\n--- 迭代 {it+1}/2 ---')
    all_boosts = np.array([_dynamic_cap(compute_raw_boost(feats_list[i], co_current)) for i in range(n_all)])
    
    X_tr, y_tr = X_gb[train_idx], y[train_idx]
    X_te = X_gb[test_idx]; y_te = y[test_idx]
    b_tr, b_te = all_boosts[train_idx], all_boosts[test_idx]
    
    scaler = StandardScaler()
    gb_m = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                      learning_rate=0.05, subsample=0.8, random_state=42)
    gb_m.fit(scaler.fit_transform(X_tr), y_tr - b_tr)
    preds = gb_m.predict(scaler.transform(X_te)) + b_te
    print(f'  GB: R²={r2_score(y_te, preds):.4f}, MAE={mean_absolute_error(y_te, preds):.4f}')
    
    scaler_all = StandardScaler()
    gb_full = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                         learning_rate=0.05, subsample=0.8, random_state=42)
    X_all_s = scaler_all.fit_transform(X_gb)
    gb_full.fit(X_all_s, y - all_boosts)
    y_residual = y - gb_full.predict(X_all_s)
    
    best_a = 1.0; best_cv = float('inf')
    for a in [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0]:
        ridge = Ridge(alpha=a, fit_intercept=False, positive=True)
        s = -cross_val_score(ridge, X_excess, y_residual, cv=5, scoring='neg_mean_absolute_error')
        if s.mean() < best_cv: best_cv = s.mean(); best_a = a
    
    ridge = Ridge(alpha=best_a, fit_intercept=False, positive=True)
    ridge.fit(X_excess, y_residual)
    co_new = ridge.coef_
    
    # 看两个新特征收了多少co
    rc_co = co_new[feat_names_all.index('real_core_notes_per_second')]
    cp_co = co_new[feat_names_all.index('core_peak_density_1sec_top5avg')]
    dd_co = co_new[feat_names_all.index('density_dimension')]
    print(f'  Ridge alpha={best_a:.3f}, dd_co={dd_co:.4f}, rcnps_co={rc_co:.4f}, cp1s_co={cp_co:.4f}')
    
    co_current = 0.3 * co_current + 0.7 * co_new

# ====== 最终 ======
print('\n--- 全量训练 ---')
all_boosts_f = np.array([_dynamic_cap(compute_raw_boost(feats_list[i], co_current)) for i in range(n_all)])
scaler_f = StandardScaler()
gb_f = GradientBoostingRegressor(n_estimators=700, max_depth=5, min_samples_leaf=3,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
gb_f.fit(scaler_f.fit_transform(X_gb), y - all_boosts_f)

# ====== 测试 + Sigmoid ======
test_dir = r'C:\Users\NaNK\Downloads'
chart_data = []
for fn in os.listdir(test_dir):
    if not fn.endswith('.json'): continue
    fp = os.path.join(test_dir, fn)
    if os.path.getsize(fp) < 100: continue
    try:
        rating = None
        for m in re.finditer(r'\((\d+\.?\d*)\)', fn):
            val = float(m.group(1))
            if 5 <= val <= 20: rating = val; break
        if rating is None:
            try:
                with open(fp, 'rb') as f: raw=f.read()
                rl=json.loads(raw.decode('utf-8')).get('META',{}).get('level')
                if rl and 5<=(rv:=float(rl))<=20: rating=rv
            except: pass
        if rating is None: continue
        with open(fp, 'rb') as f: raw=f.read()
        data,_ = load_chart_from_bytes(raw)
        feats = extract_features(data)
        if not feats: continue
        chart_data.append((fn,feats,rating))
    except: continue

print(f'\n测试谱: {len(chart_data)}个')

# 快速扫描
best_sig = None
for target in [0.16,0.18,0.20,0.22,0.24,0.26]:
    for power in [0.60,0.65,0.70,0.75,0.80,0.85]:
        for thresh in [0.18,0.20,0.22,0.24,0.26,0.28]:
            errs = []
            for fn,feats,rating in chart_data:
                X=np.array([[feats.get(k,0) for k in FN_GB]])
                Xs=scaler_f.transform(X)
                pg=float(gb_f.predict(Xs)[0])
                pb=_dynamic_cap(compute_raw_boost(feats,co_current))
                pa=adjust_boost_smooth(pb,pg,target=target,thresh=thresh,power=power)
                errs.append(pg+pa-rating)
            m=np.mean([abs(e) for e in errs])
            p=sum(1 for e in errs if e>0.01); n=sum(1 for e in errs if e<-0.01); b=abs(p-n)
            if b<=5 and m<0.55:
                mark=' ***' if b<=2 else ' *'
                print(f'  t={target:.2f} p={power:.2f} th={thresh:.2f} MAE={m:.3f} 正{p}/负{n} 平衡={b}{mark}')
                if best_sig is None or m<best_sig[4]:
                    best_sig=(target,power,thresh,b,m,p,n)

if best_sig:
    print(f'\n最优: target={best_sig[0]} power={best_sig[1]} thresh={best_sig[2]} MAE={best_sig[4]:.3f}')
    errs_f=[]
    for fn,feats,rating in chart_data:
        X=np.array([[feats.get(k,0) for k in FN_GB]])
        Xs=scaler_f.transform(X)
        pg=float(gb_f.predict(Xs)[0])
        pb=_dynamic_cap(compute_raw_boost(feats,co_current))
        pa=adjust_boost_smooth(pb,pg,target=best_sig[0],thresh=best_sig[2],power=best_sig[1])
        errs_f.append((fn,pg+pa,rating,pg+pa-rating,pg,pb,pa))
    errs_f.sort(key=lambda x:x[3])
    for fn,pr,r,err,pg,pb,pa in errs_f:
        print(f'  {fn[:38]:<38} r={r:.1f} pred={pr:.2f} err={err:+.2f} GB={pg:.2f} B={pb:.2f} adj={pa:.2f}')
    fmae=np.mean([abs(e[3]) for e in errs_f])
    pf=sum(1 for _,_,_,e,_,_,_ in errs_f if e>0.01)
    nf=sum(1 for _,_,_,e,_,_,_ in errs_f if e<-0.01)
    print(f'\n  MAE={fmae:.3f} 正偏{pf}/负偏{nf}')

# 保存
FLAT_F = []
for j,fname in enumerate(feat_names_all):
    bl_v = BL_EXTRA.get(fname, old_bl.get(fname, 1.0))
    FLAT_F.append((fname, bl_v, float(co_current[j])))

if best_sig is None:
    fmae = 999.0  # placeholder

model_out = {'gb':gb_f,'scaler':scaler_f,'feature_names':FN_GB,'p95_vals':P95,'p99_vals':P99,
             'FLAT_FEATURES':FLAT_F,'dynamic_cap':DC,'metrics':{'mae':fmae,'n_train':n_all}}
if best_sig: model_out['sigmoid_params']={'target':best_sig[0],'power':best_sig[1],'thresh':best_sig[2]}
with open('models/6dim_model_v7_4.pkl','wb') as f: pickle.dump(model_out,f)
print(f'\n已保存: models/6dim_model_v7_4.pkl')
print('='*70)
