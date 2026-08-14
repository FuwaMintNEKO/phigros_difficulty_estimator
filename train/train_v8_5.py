"""
v8.5: 耐力重构=高潮段焦点 + 删除位移 + 读谱判定线 + >32nd快音符
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

print('='*70); print('  v8.5 — 耐力=高潮段焦点 + 删除位移 + >32nd'); print('='*70)

with open('models/6dim_model_v8_4.pkl', 'rb') as f:
    v84 = pickle.load(f)
P95o = v84['p95_vals']; P99o = v84['p99_vals']
DC = v84['dynamic_cap']
FNo = v84['feature_names']

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
print(f'IN/AT: {n_all}, 难度 {labels.min():.1f}~{labels.max():.1f}')

old_co = {f[0]: f[2] for f in v84['FLAT_FEATURES']}
old_bl = {f[0]: f[1] for f in v84['FLAT_FEATURES']}

FLAT = [
    # ====== 密度 ======
    ('density_dimension',              1.0,    0.08),
    ('real_core_notes_per_second',     2.0,    0.05),
    # ====== 配置 ======
    ('stair_density',                  1.0,    old_co.get('stair_density', 0.03)),
    ('stair_speed_avg',                8.0,    old_co.get('stair_speed_avg', 0.07)),
    ('stair_complexity',               0.2,    old_co.get('stair_complexity', 0.02)),
    ('stair_chord_ratio',              0.3,    old_co.get('stair_chord_ratio', 0.0002)),
    ('trill_density',                  2.0,    old_co.get('trill_density', 0.0001)),
    ('chord_size_entropy',             0.5,    old_co.get('chord_size_entropy', 0.03)),
    ('multi_finger_3plus_events',      10.0,   old_co.get('multi_finger_3plus_events', 0.0001)),
    ('chord_alternation_rate',         0.5,    old_co.get('chord_alternation_rate', 0.12)),
    ('weighted_mf_score_per_sec',      10.0,   old_co.get('weighted_mf_score_per_sec', 0.09)),
    ('discrete_mf_ratio',              0.3,    old_co.get('discrete_mf_ratio', 0.0001)),
    ('position_entropy',               2.0,    old_co.get('position_entropy', 0.03)),
    ('position_range_used',            0.5,    old_co.get('position_range_used', 0.09)),
    ('pattern_switch_rate',            1.0,    old_co.get('pattern_switch_rate', 0.06)),
    ('direction_irregularity',         0.5,    old_co.get('direction_irregularity', 0.001)),
    ('drag_flick_ratio',               0.2,    old_co.get('drag_flick_ratio', 0.01)),
    ('avg_chord_size_poly',            2.0,    old_co.get('avg_chord_size_poly', 0.06)),
    # ====== 耐力: 高潮段焦点 + 量表补充（仅高物量生效） ======
    ('above_avg_density_mean',         4.0,    0.20),   # 高潮段平均TPS(强信号 r=0.83)
    ('total_notes',                    400.0,  0.15),   # 总物量 r=0.77，低物量(<743)不激活
    ('rest_ratio',                     0.3,    0.08),   # 休息段比例
    ('tap_burst_top5',                 0.5,    old_co.get('tap_burst_top5', 0.05)),
    # ====== 读谱: 通用特征降权 ======
    ('tempo_change_count',             50.0,   0.02),
    ('rhythm_entropy',                 2.5,    0.05),
    ('type_switch_per_sec',            0.4,    old_co.get('type_switch_per_sec', 0.07)),
    ('note_clutter_ratio',             0.05,   0.05),
    ('density_transition_mean',        0.15,   0.03),
    ('density_transition_std',         0.2,    0.05),
    ('hold_interference_index',        0.3,    0.05),
    # ====== 读谱: Phigros判定线视觉干扰 ======
    ('jline_movement_density',         50.0,   0.06),
    ('jline_rotate_density',           20.0,   0.04),
    ('jline_disappear_density',        20.0,   0.04),
    ('speed_volatility',               0.1,    0.04),
    ('above_below_cross',              0.3,    0.03),
    # ====== 高速音符（锁定co） ======
    ('fast_note_density_16th',         4.0,    0.08),
    ('fast_note_density_32nd',         2.0,    0.15),
    ('fast_note_density_24th',         1.0,    0.10),
    ('fast_note_density_48th',         0.5,    0.12),    # 48分=2倍32分速
    ('fast_note_density_64th',         0.3,    0.10),    # 64分
    ('rhythm_type_count',              3.0,    old_co.get('rhythm_type_count', 0.13)),
]

feat_names_boost = [f[0] for f in FLAT]
print(f'Boost特征数: {len(FLAT)}')

# ====== 锁定fast_note系列co，Ridge不得优化 ======
PINNED = {'fast_note_density_16th': 0.08, 'fast_note_density_32nd': 0.15,
          'fast_note_density_24th': 0.10, 'fast_note_density_48th': 0.12,
          'fast_note_density_64th': 0.10, 'total_notes': 0.15}
pinned_indices = [i for i, (n,_,_) in enumerate(FLAT) if n in PINNED]
free_indices = [i for i in range(len(FLAT)) if i not in pinned_indices]
print(f'锁定特征({len(pinned_indices)}): {[FLAT[i][0] for i in pinned_indices]}')
print(f'锁定特征({len(pinned_indices)}): {[FLAT[i][0] for i in pinned_indices]}')
print(f'自由特征({len(free_indices)}): Ridge可优化')

# === 更新 P95/P99 ===
P95 = dict(P95o); P99 = dict(P99o)
new_feats = ['cross_hand_density','burst_movement_variance',
             'jline_movement_density','jline_rotate_density','jline_disappear_density',
             'speed_volatility','above_below_cross',
             'fast_note_density_48th','fast_note_density_64th',
             'above_avg_density_mean','total_notes']
for feat in new_feats + ['density_dimension','above_avg_density_ratio','rest_ratio']:
    vals = [f.get(feat, 0) for f in feats_list]
    if vals:
        P95[feat] = float(np.percentile(vals, 95))
        P99[feat] = float(np.percentile(vals, 99))
        print(f'{feat}: P95={P95.get(feat,0):.4f} P99={P99.get(feat,0):.4f}')

def compute_excess(feats, fname, bl):
    val = feats.get(fname, 0)
    pv = P95.get(fname, 0)
    thresh = max(pv * 0.55, bl * 0.5)
    if val <= thresh: return 0.0
    excess = (val / thresh - 1.0) ** 0.70
    if val > max(P99.get(fname, 0), bl * 0.5):
        pe = (val / max(P99.get(fname, 0), bl * 0.5) - 1.0) ** 0.70
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

co_current = np.array([PINNED.get(n, c) for n, _, c in FLAT])

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
    print(f'  GB: R²={r2:.4f}, MAE={mae:.4f}')
    
    sc_all = StandardScaler()
    gb_full = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                         learning_rate=0.05, subsample=0.8, random_state=42)
    gb_full.fit(sc_all.fit_transform(X_gb), y - all_boosts)
    y_res = y - gb_full.predict(sc_all.transform(X_gb))
    
    # 扣除锁定特征的excess贡献，Ridge只优化自由特征
    X_excess_free = X_excess[:, free_indices]
    pinned_contrib = np.zeros(n_all)
    for pi in pinned_indices:
        co_pin = PINNED[FLAT[pi][0]]
        pinned_contrib += co_pin * X_excess[:, pi]
    y_res_free = y_res - pinned_contrib
    
    best_a = 5.0; best_cv = float('inf')
    for a in [0.01, 0.1, 1, 5, 10, 50, 100]:
        ridge = Ridge(alpha=a, fit_intercept=False, positive=True)
        s = -cross_val_score(ridge, X_excess_free, y_res_free, cv=5, scoring='neg_mean_absolute_error')
        if s.mean() < best_cv: best_cv = s.mean(); best_a = a
    ridge = Ridge(alpha=best_a, fit_intercept=False, positive=True)
    ridge.fit(X_excess_free, y_res_free)
    
    # 合并: 锁定co + Ridge学到的co
    co_new = np.zeros(n_boost)
    for fi, pi in enumerate(free_indices):
        co_new[pi] = ridge.coef_[fi]
    for pi in pinned_indices:
        co_new[pi] = PINNED[FLAT[pi][0]]
    
    for k in ['density_dimension','above_avg_density_mean','total_notes',
              'jline_movement_density','speed_volatility','above_below_cross',
              'rest_ratio','tap_burst_top5','tempo_change_count','hold_interference_index',
              'fast_note_density_16th','fast_note_density_32nd','fast_note_density_24th',
              'fast_note_density_48th','fast_note_density_64th']:
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
for target in [0.28, 0.30, 0.32]:
    for power in [0.65, 0.70, 0.75, 0.80]:
        for thresh in [0.20, 0.22, 0.24, 0.26]:
            for cf in [1.00, 1.04, 1.08, 1.12, 1.16]:
                errs = []
                scan_co = co_current.copy()
                for fi in free_indices:
                    scan_co[fi] = co_current[fi] * cf
                for fn, feats, rating in chart_data:
                    X = np.array([[feats.get(k,0) for k in FN]])
                    pg = float(gb_f.predict(sc_f.transform(X))[0])
                    raw = compute_raw_boost(feats, scan_co)
                    pb = _dc(raw)
                    pa = adjust_boost(pb, pg, target=target, thresh=thresh, power=power)
                    errs.append(pg + pa - rating)
                rated = [e for e, (_,_,r) in zip(errs, chart_data) if r > 0]
                if not rated: continue
                m = np.mean([abs(e) for e in rated])
                pos = sum(1 for e, (_,_,r) in zip(errs, chart_data) if r > 0 and e > 0.01)
                neg = sum(1 for e, (_,_,r) in zip(errs, chart_data) if r > 0 and e < -0.01)
                if abs(pos-neg) <= 10 and m < 0.70:
                    if best is None or m < best[5]:
                        best = (target, power, thresh, cf, abs(pos-neg), m, pos, neg)

if best:
    t, p, th, cf, bal, mae, pos, neg = best
    print(f'\n最优: t={t} p={p} th={th} cf={cf} MAE={mae:.3f} +{pos}/-{neg}')
    
    # 构建最终co数组（锁定特征不缩放）
    final_co = co_current.copy()
    for fi in free_indices:
        final_co[fi] *= cf
    
    with open('models/6dim_model_v8_4.pkl', 'rb') as f:
        v84m = pickle.load(f)
    gb_o = v84m['gb']; sc_o = v84m['scaler']; fn_o = v84m['feature_names']
    flat_o = v84m['FLAT_FEATURES']; dc_o = v84m['dynamic_cap']
    p95_o = v84m['p95_vals']; p99_o = v84m['p99_vals']
    
    key_tags = ['Final EndGame','朧月','Apollo','恋ひ恋ふ縁','怪文書','Waking',
                'ex7','ex8','Regrets','silly','Submerged','Cheerio','Chart_SP',
                'スタートリップ',
                'おぎゃり','茉子','トキ','ふたり']
    
    print(f'\n{"谱面":<25} {"定数":>5} {"v8.4":>7} {"v8.5":>7} {"Δ":>7}')
    print('-' * 58)
    
    for fn, feats, rating in chart_data:
        matched = False
        for tag in key_tags:
            if tag in fn: matched = True; break
        if not matched: continue
        
        Xo = np.array([[feats.get(k,0) for k in fn_o]])
        po_gb = float(gb_o.predict(sc_o.transform(Xo))[0])
        po_raw = 0.0
        for fname, bl, co in flat_o:
            v = feats.get(fname, 0); pv = p95_o.get(fname, 0)
            th = max(pv * 0.55, bl * 0.5)
            if v <= th: continue
            e = (v/th - 1.0) ** 0.70
            p99_v = p99_o.get(fname, 0)
            if v > max(p99_v, bl * 0.5):
                pe = (v / max(p99_v, bl * 0.5) - 1.0) ** 0.70
                e += 0.5 * max(0, pe) ** 0.70
            po_raw += co * e
        po_b = po_raw if po_raw <= dc_o['knee'] else dc_o['knee'] + (po_raw - dc_o['knee']) ** dc_o['power']
        po_a = adjust_boost(po_b, po_gb, target=0.28, thresh=0.22, power=0.75)
        po_pred = po_gb + po_a
        
        Xn = np.array([[feats.get(k,0) for k in FN]])
        pn_gb = float(gb_f.predict(sc_f.transform(Xn))[0])
        pn_raw = compute_raw_boost(feats, final_co)
        pn_b = _dc(pn_raw)
        pn_a = adjust_boost(pn_b, pn_gb, target=t, thresh=th, power=p)
        pn_pred = pn_gb + pn_a
        
        short = fn[:23]
        print(f'{short:<25} {rating:>5.1f} {po_pred:>7.2f} {pn_pred:>7.2f} {pn_pred-po_pred:>+7.2f}')

# 保存
FLAT_F = [(fname, bl, float(final_co[j])) for j, (fname, bl, _) in enumerate(FLAT)]
out = {'gb': gb_f, 'scaler': sc_f, 'feature_names': list(FN),
       'p95_vals': P95, 'p99_vals': P99, 'FLAT_FEATURES': FLAT_F, 'dynamic_cap': DC,
       'sigmoid_params': {'target': t, 'power': p, 'thresh': th}}
os.makedirs('models', exist_ok=True)
with open('models/6dim_model_v8_5.pkl', 'wb') as f: pickle.dump(out, f)
print(f'\n已保存: models/6dim_model_v8_5.pkl')
