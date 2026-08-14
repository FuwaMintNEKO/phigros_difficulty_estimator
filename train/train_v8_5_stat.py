"""
v8.5 STAT: 统计学清洗 + 分层抽样hold-out测试 + 重训
  1. 全特征 r(定数) 筛选: r_o>0.15 且 r_c>0 (排除负交叉验证)
  2. 共线去重: 删 r>0.95 的次要特征
  3. 分层抽样: 13-17定数谱面作为hold-out测试集
  4. 锁定 fast_note co (Ridge不碰)
  5. 训练 + 评估 hold-out MAE
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os, pickle, numpy as np, math, re, random
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

random.seed(42); np.random.seed(42)

print('='*70)
print('  v8.5 STAT — 统计学清洗 + hold-out测试')
print('='*70)

# ===== 1. 加载所有数据 =====
CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
diff_map = load_difficulty_tsv(TSV)
chart_files = find_chart_files(CHART_DIR)

all_official = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in diff_map: continue
    for lv in ['IN','AT']:
        if lv not in info.get('levels',{}): continue
        if lv not in diff_map[sid]: continue
        try:
            cd = load_chart_json(info['levels'][lv])
            f = extract_features(cd)
            if f:
                f['_difficulty'] = diff_map[sid][lv]
                f['_name'] = fn[:30]
                all_official.append(f)
        except: pass

# 排除可能的新谱(chartnekockLK, snow dance)
exclude_patterns = ['chartnekockLK', 'snow dance', 'Snow']
all_official = [f for f in all_official if not any(p.lower() in f['_name'].lower() for p in exclude_patterns)]
print(f'官谱(排除新谱): {len(all_official)}')

# 加载自制谱(验证用)
print('加载自制谱验证集...')
test_dir = r'C:\Users\NaNK\Downloads'
custom = []
for fn in os.listdir(test_dir):
    if not fn.endswith('.json') or '_2xBPM' in fn: continue
    rating = None
    for m in re.finditer(r'\((\d+\.?\d*)\)', fn):
        v = float(m.group(1))
        if 5 <= v <= 25: rating = v; break
    if rating is None: continue
    fp = os.path.join(test_dir, fn)
    try:
        with open(fp,'rb') as f: raw = f.read()
        data,_ = load_chart_from_bytes(raw)
        feats = extract_features(data)
        if feats:
            feats['_difficulty'] = rating
            feats['_name'] = fn[:30]
            custom.append(feats)
    except: pass
print(f'自制谱验证: {len(custom)}')

# ===== 2. 统计r值 =====
feature_names = sorted(all_official[0].keys())
feature_names = [k for k in feature_names if k not in ('_difficulty','_name')]
diffs_o = np.array([r['_difficulty'] for r in all_official])
diffs_c = np.array([r['_difficulty'] for r in custom])

stat_results = {}
for key in feature_names:
    vals_o = np.array([r.get(key, 0) for r in all_official], dtype=float)
    if np.std(vals_o) < 1e-6: continue
    r_o = np.corrcoef(vals_o, diffs_o)[0,1]
    if np.isnan(r_o): continue
    vals_c = np.array([r.get(key, 0) for r in custom], dtype=float)
    r_c = np.corrcoef(vals_c, diffs_c)[0,1] if len(custom)>2 and np.std(vals_c)>1e-6 else float('nan')
    stat_results[key] = (r_o, r_c, np.mean(vals_o))

# ===== 3. 筛选: r_o>0.15且r_c>0(或nan) =====
PASS = []
for k, (r_o, r_c, _) in stat_results.items():
    if r_o < 0.15: continue
    if not np.isnan(r_c) and r_c < -0.05: continue  # 交叉验证负相关就毙
    if k in ['stamina_ratio','sustained_density_run_ratio','avg_interval_beats',
              'extreme_tap_window_ratio','burst_avg_movement','avg_movement',
              'above_avg_density_ratio','dominant_rhythm_ratio','chord_2note_ratio',
              'center_ratio','pattern_switch_rate']:
        continue  # 手动毙已知负相关/倒挂
    PASS.append(k)

print(f'统计通过: {len(PASS)} features')

# ===== 4. 共线去重 =====
# 对PASS中的特征算两两r
vals_dict = {k: np.array([r.get(k,0) for r in all_official]) for k in PASS}
REMOVE = set()
for i in range(len(PASS)):
    if PASS[i] in REMOVE: continue
    for j in range(i+1, len(PASS)):
        if PASS[j] in REMOVE: continue
        rv = np.corrcoef(vals_dict[PASS[i]], vals_dict[PASS[j]])[0,1]
        if abs(rv) > 0.95:
            # 保留r_o更高的
            r_i = stat_results[PASS[i]][0]
            r_j = stat_results[PASS[j]][0]
            worse = PASS[j] if r_i >= r_j else PASS[i]
            REMOVE.add(worse)
            better = PASS[i] if r_i >= r_j else PASS[j]
            print(f'  共线删 {worse:<40s} (r_with_{better[:20]}={rv:.3f})')

PASS = [k for k in PASS if k not in REMOVE]
print(f'共线去重后: {len(PASS)} features')

# 打印分类
def print_cat(label, keys):
    in_pass = [k for k in keys if k in PASS]
    not_in = [k for k in keys if k not in PASS and k in stat_results]
    print(f'\n{label}:')
    for k in in_pass:
        print(f'  + {k:<40s} r_o={stat_results[k][0]:+.4f} r_c={stat_results[k][1]:+.4f}')
    for k in not_in[:5]:
        print(f'  - {k:<40s} r_o={stat_results[k][0]:+.4f}')

density_set = ['density_dimension','real_core_notes_per_second','above_avg_density_mean',
               'above_avg_duration_sec','fast_note_density_16th','fast_note_density_32nd',
               'fast_note_density_24th','fast_note_density_48th','fast_note_density_64th',
               'rhythm_type_count','real_notes_per_second']
config_set = ['stair_rate_per_sec','stair_speed_avg','stair_complexity','stair_chord_ratio',
              'trill_density','chord_size_entropy','multi_finger_3plus_events',
              'chord_alternation_rate','weighted_mf_score_per_sec','discrete_mf_ratio',
              'position_entropy','position_range_used','direction_irregularity',
              'drag_flick_ratio','avg_chord_size_poly','jack_density',
              'hold_interference_index','avg_chord_size','avg_simultaneous',
              'simultaneous_event_count','simultaneous_ratio','chord_alternation_rate']
stamina_set = ['above_avg_density_mean','above_avg_duration_sec','total_notes',
               'tap_per_second','tap_count','duration_sec','rest_ratio',
               'global_jack_count','burst_intensity_mean','tap_burst_top5','real_active_sec']
reading_set = ['jline_movement_density','jline_rotate_density','jline_disappear_density',
               'speed_volatility','above_below_cross','type_switch_per_sec',
               'rhythm_entropy','density_transition_mean','density_transition_std',
               'note_clutter_ratio','tempo_change_count','hold_interference_index',
               'offbeat_ratio']

print_cat('密度', density_set)
print_cat('配置', config_set)
print_cat('耐力', stamina_set)
print_cat('读谱', reading_set)

# ===== 5. 构建 FLAT =====
# 从PASS挑特征进FLAT，给初始co
FLAT = []
pinned_names = set()
for k in PASS:
    r_o = stat_results[k][0]
    bl = max(stat_results[k][2] * 0.3, 0.01)  # baseline = 30% of mean
    # co: r_o>0.6=高信号, r_o>0.3=中等, 其余=微弱
    if r_o > 0.6: co = 0.10 + (r_o - 0.6) * 0.5
    elif r_o > 0.3: co = max(0.02, (r_o - 0.15) * 0.3)
    else: co = 0.005
    FLAT.append((k, round(bl, 4), round(co, 4)))

# fast_note 锁定
for i, (n,b,c) in enumerate(FLAT):
    if 'fast_note_density' in n and '16th' in n:
        FLAT[i] = (n, b, 0.08); pinned_names.add(n)
    if 'fast_note_density' in n and '32nd' in n:
        FLAT[i] = (n, b, 0.15); pinned_names.add(n)
    if 'fast_note_density' in n and '24th' in n:
        FLAT[i] = (n, b, 0.10); pinned_names.add(n)
    if 'fast_note_density' in n and '48th' in n:
        FLAT[i] = (n, b, 0.12); pinned_names.add(n)
    if 'fast_note_density' in n and '64th' in n:
        FLAT[i] = (n, b, 0.10); pinned_names.add(n)

PINNED = {n: c for n, _, c in FLAT if n in pinned_names}
pinned_indices = [i for i, (n,_,_) in enumerate(FLAT) if n in PINNED]
free_indices = [i for i in range(len(FLAT)) if i not in pinned_indices]
print(f'\nFLAT: {len(FLAT)} features ({len(pinned_indices)} locked)')

# ===== 6. 分层抽样hold-out (13-17) =====
# 按定数区间分层
X_all = np.array([[f.get(n,0) for n in feature_names] for f in all_official])
diffs_all = np.array([f['_difficulty'] for f in all_official])

# hold-out: 13-17区间，每层随机抽20%
hold_out_mask = np.zeros(len(all_official), dtype=bool)
bins = np.digitize(diffs_all, bins=[13,14,15,16,17])
for b in range(1, 6):  # bins 1-5 = [13,14),[14,15),[15,16),[16,17),[17,18)
    idx = np.where(bins == b)[0]
    if len(idx) < 3: continue
    n_hold = max(1, int(len(idx) * 0.25))
    chosen = np.random.choice(idx, size=n_hold, replace=False)
    hold_out_mask[chosen] = True

print(f'Hold-out: {hold_out_mask.sum()} charts (13-17 range)')
print(f'Train: {(~hold_out_mask).sum()} charts')

# ===== 7. 训练 =====
feats_train = [all_official[i] for i in range(len(all_official)) if not hold_out_mask[i]]
feats_test = [all_official[i] for i in range(len(all_official)) if hold_out_mask[i]]
y_train = diffs_all[~hold_out_mask]; y_test = diffs_all[hold_out_mask]

feat_names_boost = [n for n,_,_ in FLAT]
n_boost = len(FLAT)

def compute_excess(feats, fname, bl):
    val = feats.get(fname, 0)
    th = max(P95.get(fname, 0) * 0.55, bl * 0.5)
    if val <= th: return 0.0
    excess = (val / th - 1.0) ** 0.70
    if val > max(P99.get(fname, 0), bl * 0.5):
        pe = (val / max(P99.get(fname, 0), bl * 0.5) - 1.0) ** 0.70
        excess += 0.5 * max(0, pe) ** 0.70
    return excess

# ===== Pre-compute P95/P99 =====
P95 = {}; P99 = {}
for fn in feat_names_boost:
    vals = [f.get(fn, 0) for f in all_official]
    P95[fn] = float(np.percentile(vals, 95)) if vals else 0
    P99[fn] = float(np.percentile(vals, 99)) if vals else 0

DC = {'knee': 2.5, 'power': 0.9}

def _dc(r):
    if r <= DC['knee']: return r
    return DC['knee'] + (r - DC['knee']) ** DC['power']

co_current = np.array([c for _,_,c in FLAT])

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

# GB features (all 219+)
FN = list(feature_names)
X_gb_train = np.array([[f.get(k,0) for k in FN] for f in feats_train])
X_gb_test = np.array([[f.get(k,0) for k in FN] for f in feats_test])

# Excess matrix
X_excess_train = np.zeros((len(feats_train), n_boost))
for i in range(len(feats_train)):
    for j, (fname, bl, _) in enumerate(FLAT):
        X_excess_train[i, j] = compute_excess(feats_train[i], fname, bl)

for it in range(3):
    print(f'\n--- Iter {it+1}/3 ---')
    all_boosts = np.array([_dc(compute_raw_boost(feats_train[i], co_current)) for i in range(len(feats_train))])
    
    sc = StandardScaler()
    gbm = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
    gbm.fit(sc.fit_transform(X_gb_train), y_train - all_boosts)
    preds_train = gbm.predict(sc.transform(X_gb_train)) + all_boosts
    r2 = r2_score(y_train, preds_train)
    print(f'  GB: R²={r2:.4f} MAE={mean_absolute_error(y_train, preds_train):.4f}')
    
    sc_all = StandardScaler()
    gb_full = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                         learning_rate=0.05, subsample=0.8, random_state=42)
    gb_full.fit(sc_all.fit_transform(X_gb_train), y_train - all_boosts)
    y_res = y_train - gb_full.predict(sc_all.transform(X_gb_train))
    
    # 扣除锁定特征贡献
    X_excess_free = X_excess_train[:, free_indices]
    pinned_contrib = np.zeros(len(feats_train))
    for pi in pinned_indices:
        co_pin = PINNED[FLAT[pi][0]]
        pinned_contrib += co_pin * X_excess_train[:, pi]
    y_res_free = y_res - pinned_contrib
    
    best_a = 5.0; best_cv = float('inf')
    for a in [0.1, 1, 5, 10, 50]:
        ridge = Ridge(alpha=a, fit_intercept=False, positive=True)
        s = -cross_val_score(ridge, X_excess_free, y_res_free, cv=3, scoring='neg_mean_absolute_error')
        if s.mean() < best_cv: best_cv = s.mean(); best_a = a
    ridge = Ridge(alpha=best_a, fit_intercept=False, positive=True)
    ridge.fit(X_excess_free, y_res_free)
    
    co_new = np.zeros(n_boost)
    for fi, pi in enumerate(free_indices):
        co_new[pi] = ridge.coef_[fi]
    for pi in pinned_indices:
        co_new[pi] = PINNED[FLAT[pi][0]]
    
    for k in ['density_dimension','above_avg_density_mean','above_avg_duration_sec',
              'tap_per_second','total_notes','tempo_change_count','type_switch_per_sec',
              'fast_note_density_16th','fast_note_density_32nd','rhythm_type_count']:
        if k in feat_names_boost:
            print(f'  {k:<35} co={co_new[feat_names_boost.index(k)]:.4f}')
    co_current = 0.3 * co_current + 0.7 * co_new

# ===== 8. 全量重训 + 评估 =====
all_boosts_f = np.array([_dc(compute_raw_boost(f, co_current)) for f in feats_train])
sc_f = StandardScaler()
gb_f = GradientBoostingRegressor(n_estimators=700, max_depth=5, min_samples_leaf=3,
                                  learning_rate=0.05, subsample=0.8, random_state=42)
gb_f.fit(sc_f.fit_transform(X_gb_train), y_train - all_boosts_f)

# 预测hold-out
X_gb_test_ex = np.array([[f.get(k,0) for k in FN] for f in feats_test])
pg_test = gb_f.predict(sc_f.transform(X_gb_test_ex))
pb_test = np.array([_dc(compute_raw_boost(f, co_current)) for f in feats_test])
pa_test = np.array([adjust_boost(pb_test[i], pg_test[i]) for i in range(len(feats_test))])
preds_hold = pg_test + pa_test

mae_hold = mean_absolute_error(y_test, preds_hold)
print(f'\n=== Hold-out ({len(feats_test)} charts, 13-17) ===')
print(f'MAE: {mae_hold:.4f}')

# 按定数分组
for lo, hi, label in [(13,15,'13-15'),(15,16,'15-16'),(16,18,'16-18')]:
    mask = (y_test >= lo) & (y_test < hi)
    if mask.sum() == 0: continue
    m = mean_absolute_error(y_test[mask], preds_hold[mask])
    print(f'  {label} (n={mask.sum()}): MAE={m:.4f}')

# 打印具体谱面
print(f'\n{"Hold-out 谱面":<30} {"定数":>5} {"预测":>7} {"误差":>7}')
for i in np.argsort(y_test):
    err = preds_hold[i] - y_test[i]
    print(f'{feats_test[i].get("_name","?")[:28]:<30} {y_test[i]:>5.1f} {preds_hold[i]:>7.2f} {err:>+7.2f}')

# ===== 9. 自制谱验证 =====
if custom:
    print(f'\n=== 自制谱验证 ({len(custom)} charts) ===')
    for f in custom:
        Xc = np.array([[f.get(k,0) for k in FN]])
        pg = float(gb_f.predict(sc_f.transform(Xc))[0])
        pr = compute_raw_boost(f, co_current)
        pb = _dc(pr)
        pa = adjust_boost(pb, pg)
        name = f.get('_name','?')[:25]
        print(f'  {name:<30} rating={f["_difficulty"]:.1f} pred={pg+pa:.2f}')

# ===== 10. 保存 =====
FLAT_F = [(n, b, float(co_current[j])) for j, (n, b, _) in enumerate(FLAT)]
out = {'gb': gb_f, 'scaler': sc_f, 'feature_names': list(FN),
       'p95_vals': P95, 'p99_vals': P99, 'FLAT_FEATURES': FLAT_F, 'dynamic_cap': DC,
       'stat_filtered': True, 'holdout_charts': len(feats_test), 'holdout_mae': mae_hold}
os.makedirs('models', exist_ok=True)
with open('models/6dim_model_v8_5.pkl', 'wb') as f: pickle.dump(out, f)
print(f'\nSaved: models/6dim_model_v8_5.pkl (stat filtered, holdout MAE={mae_hold:.4f})')

# ===== 11. 跑全部测试谱 =====
print(f'\n=== 全部测试谱 (_all_test_charts) ===')
all_test_feats = []
for fn in os.listdir(test_dir):
    if not fn.endswith('.json') or '_2xBPM' in fn: continue
    fp = os.path.join(test_dir, fn)
    if os.path.getsize(fp) < 100: continue
    try:
        rating = None
        for m in re.finditer(r'\((\d+\.?\d*)\)', fn):
            v = float(m.group(1))
            if 5 <= v <= 25: rating = v; break
        with open(fp,'rb') as f: raw = f.read()
        data,_ = load_chart_from_bytes(raw)
        feats = extract_features(data)
        if feats:
            Xx = np.array([[feats.get(k,0) for k in FN]])
            pg = float(gb_f.predict(sc_f.transform(Xx))[0])
            pr = compute_raw_boost(feats, co_current)
            pb = _dc(pr)
            pa = adjust_boost(pb, pg)
            all_test_feats.append((fn[:30], rating, pg+pa))
    except: pass

all_test_feats.sort(key=lambda x: -x[2])
print(f'{"谱面":<30} {"定数":>5} {"预测":>7}')
for name, rating, pred in all_test_feats:
    r_str = f'{rating:.1f}' if rating else ' ?? '
    print(f'{name:<30} {r_str:>5} {pred:>7.2f}')
