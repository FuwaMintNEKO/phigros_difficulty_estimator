"""
v8.1: BPM timeline积分修复 + 新配置特征

新增4个配置特征(替代5个旧弱特征):
  fast_note_density_16th: 同线≤16分音次数/秒 (corr=0.84)
  mf_weight_per_sec: 多押梯形权重/秒 (corr=0.70)
  avg_chord_size_poly: 多押平均size (corr=0.59)
  rhythm_type_count: 唯一节奏类型数 (corr=0.61)

删除(弱/冗余):
  jack_density, sim_pos_spread_mean, position_cluster_count,
  track_deviation_score, drag_flick_ratio

配置维度: 20 → 19特征, 但每个都更强
"""
import sys, json, os, pickle, numpy as np, math, re
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, '.')
from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print('='*70); print('  v8.1 — BPM timeline修复 + 新配置特征'); print('='*70)

# ====== 加载v7.3获取P95/P99 ======
with open('models/6dim_model_v7_3.pkl', 'rb') as f:
    v73 = pickle.load(f)
P95_old = v73['p95_vals']; P99_old = v73['p99_vals']
DC = v73['dynamic_cap']; FLAT_OLD = v73['FLAT_FEATURES']
FN_OLD = v73['feature_names']

# ====== 加载官谱IN/AT ======
CHART_DIR = r'D:\迅雷下载\Phigros_Resource-master\Phigros_Resource-master\chart'
DIFFICULTY_TSV = r'D:\迅雷下载\Phigros_Resource-master\Phigros_Resource-master\info\difficulty.tsv'
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
        feats = extract_features(cd)  # v8.0 config features now computed natively
        if feats:
            feats_list.append(feats)
            labels.append(item['difficulty'])
    except: pass

n_all = len(feats_list); labels = np.array(labels)
print(f'\nIN/AT: {n_all}, 难度 {labels.min():.1f}~{labels.max():.1f}')

# ====== 新特征的P95/P99 ======
# 所有新特征（计算阶段需要全算，boost只选3个）
ALL_NEW_FEATS = ['fast_note_density_16th', 'fast_note_density_32nd',
                 'rhythm_type_count', 'mf_weight_per_sec', 'avg_chord_size_poly', 'max_chord_size_poly']

P95_new = {}; P99_new = {}
for k in ALL_NEW_FEATS:
    vals = [f.get(k, 0) for f in feats_list]
    P95_new[k] = float(np.percentile(vals, 95))
    P99_new[k] = float(np.percentile(vals, 99))

P95 = {**P95_old, **P95_new}
P99 = {**P99_old, **P99_new}
for k in ALL_NEW_FEATS:
    print(f'  {k:<30} P95={P95_new[k]:.2f} P99={P99_new[k]:.2f}')

# ====== 构建新FLAT_FEATURES ======
# 删除的旧特征
BOOST_REMOVE = {'jack_density', 'sim_pos_spread_mean', 'position_cluster_count',
                'track_deviation_score', 'drag_flick_ratio'}

# 旧co (用于种子, 删除的略过)
old_co = {f[0]: f[2] for f in FLAT_OLD}
old_bl = {f[0]: f[1] for f in FLAT_OLD}

FLAT_NEW = []
for fname, bl, co in FLAT_OLD:
    if fname in BOOST_REMOVE:
        continue
    FLAT_NEW.append((fname, bl, co))

# 加新特征 — 只保留最强的3个
NEW_FEAT_NAMES = ['fast_note_density_16th', 'avg_chord_size_poly', 'rhythm_type_count']
NEW_BOOST = [
    ('fast_note_density_16th', 4.0, 0.25),    # 快速音符密度 (corr=0.84)
    ('avg_chord_size_poly', 2.0, 0.20),        # 多押平均size (corr=0.59)
    ('rhythm_type_count', 3.0, 0.12),          # 节奏多样性 (corr=0.61)
]
FLAT_NEW.extend(NEW_BOOST)
for fn, bl, co in NEW_BOOST:
    old_bl[fn] = bl; old_co[fn] = co

# 新特征只加入boost, 不加入GB (GB用v7.3的219特征)
FN_NEW = list(FN_OLD)  # ← 不改GB!
print(f'Boost: {len(FLAT_OLD)} → {len(FLAT_NEW)} (删除{BOOST_REMOVE}, 新增{NEW_FEAT_NAMES})')
print(f'GB: {len(FN_NEW)}特征 (不变, 新特征仅boost)')

# ====== excess计算 ======
feat_names_boost = [f[0] for f in FLAT_NEW]
n_boost = len(FLAT_NEW)

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
    for j, (fname, bl, _) in enumerate(FLAT_NEW):
        X_excess[i, j] = compute_excess(feats_list[i], fname, bl)

# GB特征: 不加入新特征, 仅v7.3的219
FN_NEW = list(FN_OLD)  # ← 不改GB!
X_gb = np.array([[f.get(n, 0) for n in FN_NEW] for f in feats_list])
y = labels.copy()

def _dc(r):
    if r <= DC['knee']: return r
    return DC['knee'] + (r - DC['knee']) ** DC['power']

co_current = np.array([old_co.get(fn, 0) for fn in feat_names_boost])

def compute_raw_boost(feats, co_arr):
    raw = 0.0
    for j, (fname, bl, _) in enumerate(FLAT_NEW):
        raw += co_arr[j] * compute_excess(feats, fname, bl)
    return raw

def adjust_boost(boost, gb_val, target=0.24, thresh=0.24, power=0.70):
    if boost < 2.0 or gb_val <= 0: return boost
    r = boost / gb_val; e = target * gb_val
    a = e * ((boost / e) ** power)
    w = 1 / (1 + math.exp(-25 * (r - thresh)))
    return (1 - w) * boost + w * a

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
bins = np.digitize(y, bins=[12,13,14,15,16,17,18])
train_idx, test_idx = next(sss.split(X_gb, bins))

# ====== 迭代训练 (3轮) ======
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
    imp = dict(zip(FN_NEW, gbm.feature_importances_))
    top5 = sorted(imp.items(), key=lambda x: -x[1])[:5]
    print(f'  GB: R²={r2:.4f}, MAE={mae:.4f}')
    print(f'  Top5: {" ".join(f"{k}={v:.3f}" for k,v in top5)}')
    
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
    
    # 显示新特征co
    for k in NEW_FEAT_NAMES:
        if k in feat_names_boost:
            idx = feat_names_boost.index(k)
            print(f'  {k} co={co_new[idx]:.4f}')
    
    co_current = 0.3 * co_current + 0.7 * co_new

# ====== 全量 ======
print('\n--- 全量训练 ---')
all_boosts_f = np.array([_dc(compute_raw_boost(feats_list[i], co_current)) for i in range(n_all)])
sc_f = StandardScaler()
gb_f = GradientBoostingRegressor(n_estimators=700, max_depth=5, min_samples_leaf=3,
                                  learning_rate=0.05, subsample=0.8, random_state=42)
gb_f.fit(sc_f.fit_transform(X_gb), y - all_boosts_f)
imp_f = dict(zip(FN_NEW, gb_f.feature_importances_))
print(f'  GB total_notes: {imp_f.get("total_notes",0):.4f}')

# ====== 测试谱 ======
from unified_parser import load_chart_from_bytes

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
        if rating is None: continue
        with open(fp, 'rb') as f: raw = f.read()
        data, _ = load_chart_from_bytes(raw)
        feats = extract_features(data)
        if feats:
            chart_data.append((fn, feats, rating))
    except: continue
print(f'\n测试谱: {len(chart_data)}')

# Sigmoid扫描
best_sig = None
for target in [0.18, 0.20, 0.22, 0.24, 0.26]:
    for power in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        for thresh in [0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
            errs = []
            for fn, feats, rating in chart_data:
                X = np.array([[feats.get(k,0) for k in FN_NEW]])
                pg = float(gb_f.predict(sc_f.transform(X))[0])
                pb = _dc(compute_raw_boost(feats, co_current))
                pa = adjust_boost(pb, pg, target=target, thresh=thresh, power=power)
                errs.append(pg + pa - rating)
            m = np.mean([abs(e) for e in errs])
            p = sum(1 for e in errs if e > 0.01); n = sum(1 for e in errs if e < -0.01)
            b = abs(p - n)
            if b <= 6 and m < 0.60:
                if best_sig is None or m < best_sig[4]:
                    best_sig = (target, power, thresh, b, m, p, n)

if best_sig:
    print(f'\n最优: t={best_sig[0]} p={best_sig[1]} th={best_sig[2]} MAE={best_sig[4]:.3f}')
    
    # v7.3对比
    with open('models/6dim_model_v7_3.pkl', 'rb') as f:
        v73m = pickle.load(f)
    gb_o = v73m['gb']; sc_o = v73m['scaler']; fn_o = v73m['feature_names']
    flat_o = v73m['FLAT_FEATURES']; dc_o = v73m['dynamic_cap']
    P95o = v73m['p95_vals']; P99o = v73m['p99_vals']
    
    key_tags = ['Final EndGame', '朧月', 'Apollo', '恋ひ恋ふ縁', '怪文書', 'Waking', 'Submerged', 'EndGame']
    
    print(f'\n{"谱面":<15} {"定数":>5} {"v7.3":>7} {"v8.0":>7} {"Δpred":>7} {"ΔGB":>7}')
    print('-' * 56)
    
    for fn, feats, rating in chart_data:
        short = fn[:15]
        matched = False
        for tag in key_tags:
            if tag in fn: matched = True; break
        if not matched: continue
        
        # v7.3
        Xo = np.array([[feats.get(k,0) for k in fn_o]])
        po_gb = float(gb_o.predict(sc_o.transform(Xo))[0])
        po_raw = 0.0
        for fname, bl, co in flat_o:
            v = feats.get(fname, 0); pv = P95o.get(fname, 0)
            th = max(pv * 0.55, bl * 0.5)
            if v <= th: continue
            e = (v/th - 1.0) ** 0.70
            if v > max(P99o.get(fname,0), bl * 0.5):
                pe = (v / max(P99o.get(fname,0), bl * 0.5) - 1.0)
                e += 0.5 * max(0, pe) ** 0.70
            po_raw += co * e
        po_b = po_raw if po_raw <= dc_o['knee'] else dc_o['knee'] + (po_raw - dc_o['knee']) ** dc_o['power']
        po_a = adjust_boost(po_b, po_gb)
        po_pred = po_gb + po_a
        
        # v8.0
        Xn = np.array([[feats.get(k,0) for k in FN_NEW]])
        pn_gb = float(gb_f.predict(sc_f.transform(Xn))[0])
        pn_b = _dc(compute_raw_boost(feats, co_current))
        pn_a = adjust_boost(pn_b, pn_gb, target=best_sig[0], thresh=best_sig[2], power=best_sig[1])
        pn_pred = pn_gb + pn_a
        
        # 用新特征解释 (恋ひ恋ふ縁特别关注)
        notes = []
        if '恋' in fn:
            notes.append(f'16th/s={feats.get("fast_note_density_16th",0):.1f}')
            notes.append(f'mf/s={feats.get("mf_weight_per_sec",0):.1f}')
            notes.append(f'avgChd={feats.get("avg_chord_size_poly",0):.2f}')
        
        info = '  [' + ' '.join(notes) + ']' if notes else ''
        print(f'{short:<15} {rating:>5.1f} {po_pred:>7.2f} {pn_pred:>7.2f} {pn_pred-po_pred:>+7.2f} {pn_gb-po_gb:>+7.2f}{info}')

# 保存
FLAT_F = [(fname, old_bl.get(fname, 1.0), float(co_current[j])) for j, fname in enumerate(feat_names_boost)]
out = {'gb': gb_f, 'scaler': sc_f, 'feature_names': FN_NEW,
       'p95_vals': P95, 'p99_vals': P99, 'FLAT_FEATURES': FLAT_F, 'dynamic_cap': DC}
if best_sig: out['sigmoid_params'] = {'target': best_sig[0], 'power': best_sig[1], 'thresh': best_sig[2]}
os.makedirs('models', exist_ok=True)
with open('models/6dim_model_v8_1.pkl', 'wb') as f: pickle.dump(out, f)
print(f'\n已保存: models/6dim_model_v8_1.pkl\n{"="*70}')
