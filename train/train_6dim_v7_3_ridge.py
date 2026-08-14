"""
v7.3 权重优化 — 数据驱动法

核心理念:
  1. 对每个feature, 从官谱计算超出阈值的贡献(excess^0.70)
  2. 用Ridge回归拟合 GB_error = sum(co_i * excess_i)
  3. 交叉验证选最佳正则化
  4. 联合优化sigmoid参数
  5. 最终重训GB模型
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, json, os, pickle, numpy as np, math, re, itertools
from collections import defaultdict
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, '.')
from feature_extractor import extract_features, collect_all_notes
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from unified_parser import load_chart_from_bytes

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ====== 加载v7.2模型取P95/P99 ======
with open('models/6dim_model_v7_2.pkl', 'rb') as f:
    v7m = pickle.load(f)
    
P95 = v7m['p95_vals']
P99 = v7m['p99_vals']
FLAT_ORIG = v7m['FLAT_FEATURES']
FN_GB = v7m['feature_names']

# 提取特征名列表(只保留有co的)
feat_names = [f[0] for f in FLAT_ORIG]
baselines = {f[0]: f[1] for f in FLAT_ORIG}

print('=' * 70)
print('  v7.3 — Ridge数据驱动权重优化')
print('=' * 70)

# ====== 加载全部官谱训练数据 ======
CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)
all_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    for lv in ['EZ','HD','IN','AT']:
        if lv in info['levels'] and lv in diffs:
            all_items.append({'folder':fn,'filepath':info['levels'][lv],'difficulty':diffs[lv],'level':lv})

print(f'官谱: {len(all_items)}')

# ====== 提取所有特征 ======
feats_list, labels, names_list, levels_list = [], [], [], []
for i, item in enumerate(all_items):
    try:
        cd = load_chart_json(item['filepath'])
        feats = extract_features(cd)
        if feats:
            feats_list.append(feats)
            labels.append(item['difficulty'])
            names_list.append(item['folder'])
            levels_list.append(item['level'])
    except: pass

n_total = len(feats_list)
labels = np.array(labels)
print(f'  提取: {n_total}, 难度范围 {labels.min():.1f}~{labels.max():.1f}')

# ====== 对每个feature计算excess贡献 ======
# excess = (val/thresh - 1) ** 0.70  if val > thresh
# 额外: 如果 val > P99, 加0.5 * (val/P99 - 1) ** 0.70

def compute_feature_excess(feats, fname, bl):
    val = feats.get(fname, 0)
    pv = P95.get(fname, 0)
    thresh = max(pv * 0.55, bl * 0.5)
    if val <= thresh:
        return 0.0
    excess = (val / thresh - 1.0) ** 0.70
    if val > max(P99.get(fname, 0), bl * 0.5):
        p99_excess = (val / max(P99.get(fname, 0), bl * 0.5) - 1.0)
        excess += 0.5 * max(0, p99_excess) ** 0.70
    return excess

# 构建excess矩阵 X_excess: [n_samples, n_features]
n_feats = len(feat_names)
X_excess = np.zeros((n_total, n_feats))
for i in range(n_total):
    for j, (fname, bl, _) in enumerate(FLAT_ORIG):
        X_excess[i, j] = compute_feature_excess(feats_list[i], fname, bl)

print(f'  Excess非零特征数: {np.count_nonzero(X_excess.sum(axis=0))}/{n_feats}')

# ====== 先跑基础GB获取基线残差 ======
# (用官谱全量训练一个v7.2的GB作为参照)
X_gb = np.array([[f.get(n,0) for n in FN_GB] for f in feats_list])
y = labels.copy()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
bins = np.digitize(y, bins=[0,5,7,9,11,13,14,15,16,16.5,17,18])
train_idx, test_idx = next(sss.split(X_gb, bins))

# 先用固定boost获取基线
def compute_raw_boost(feats, co_array):
    """co_array是[n_feats]数组, 对应FLAT_ORIG中的每个特征"""
    raw = 0.0
    for j, (fname, bl, _) in enumerate(FLAT_ORIG):
        excess = compute_feature_excess(feats, fname, bl)
        raw += co_array[j] * excess
    return raw

# 当前co值
co_current = np.array([f[2] for f in FLAT_ORIG])

# 先用当前co算boost
all_boosts_current = np.array([compute_raw_boost(feats_list[i], co_current) for i in range(n_total)])

# Dynamic cap
DC = {'knee': 1.0, 'power': 0.90}
def _dynamic_cap(raw):
    if raw <= DC['knee']: return raw
    return DC['knee'] + (raw - DC['knee']) ** DC['power']

all_boosts_capped = np.array([_dynamic_cap(b) for b in all_boosts_current])

# Fit GB on residuals
scaler_gb = StandardScaler()
X_tr_s = scaler_gb.fit_transform(X_gb[train_idx])
X_te_s = scaler_gb.transform(X_gb[test_idx])

gb_ref = GradientBoostingRegressor(n_estimators=700, max_depth=5, min_samples_leaf=3,
                                    learning_rate=0.05, subsample=0.8, random_state=42)
gb_ref.fit(X_tr_s, y[train_idx] - all_boosts_capped[train_idx])
y_pred_ref = gb_ref.predict(X_te_s) + all_boosts_capped[test_idx]
print(f'  基线GB: R²={r2_score(y[test_idx], y_pred_ref):.4f}, MAE={mean_absolute_error(y[test_idx], y_pred_ref):.4f}')

# 全量GB
X_all_s = scaler_gb.fit_transform(X_gb)
y_residual = y - all_boosts_capped
gb_full_ref = GradientBoostingRegressor(n_estimators=700, max_depth=5, min_samples_leaf=3,
                                         learning_rate=0.05, subsample=0.8, random_state=42)
gb_full_ref.fit(X_all_s, y_residual)

# ====== 5折交叉验证Ridge回归 ======
print('\n--- Ridge回归: 学习最优co (5折CV) ---')

# 目标: y_residual = GB_error (rating - GB_predict, 没加boost)
y_residual_full = y - gb_full_ref.predict(X_all_s)

# 对于Ridge, 我们想要: co · X_excess ≈ y_residual_full
# Ridge强制 co >= 0 或者我们后处理clip

# 用RidgeCV搜索最优alpha
kf = KFold(n_splits=5, shuffle=True, random_state=42)
alphas = np.logspace(-3, 2, 30)

best_alpha = None
best_co = None
best_cv_score = float('inf')

# 也尝试不同的GB重训
for alpha in alphas:
    ridge = Ridge(alpha=alpha, fit_intercept=False, positive=True)
    
    cv_errors = []
    for fold_train, fold_val in kf.split(X_excess):
        # 在fold_train上先重训GB(用fold_train的数据)
        X_gb_tr_fold = X_gb[fold_train]
        y_tr_fold = y[fold_train]
        X_gb_val_fold = X_gb[fold_val]
        y_val_fold = y[fold_val]
        
        # 先用网格近似找最优co
        ridge.fit(X_excess[fold_train], y_tr_fold - np.zeros_like(y_tr_fold))
        co_fold = ridge.coef_
        
        # 算boost
        boosts_tr_fold = np.array([compute_raw_boost(feats_list[i], co_fold) for i in fold_train])
        boosts_tr_fold = np.array([_dynamic_cap(b) for b in boosts_tr_fold])
        boosts_val_fold = np.array([compute_raw_boost(feats_list[i], co_fold) for i in fold_val])
        boosts_val_fold = np.array([_dynamic_cap(b) for b in boosts_val_fold])
        
        # 重训GB
        sc_fold = StandardScaler()
        gb_fold = GradientBoostingRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                             learning_rate=0.08, subsample=0.8, random_state=42)
        gb_fold.fit(sc_fold.fit_transform(X_gb_tr_fold), y_tr_fold - boosts_tr_fold)
        
        preds = gb_fold.predict(sc_fold.transform(X_gb_val_fold)) + boosts_val_fold
        mae_fold = mean_absolute_error(y_val_fold, preds)
        cv_errors.append(mae_fold)
    
    mean_cv = np.mean(cv_errors)
    if mean_cv < best_cv_score:
        best_cv_score = mean_cv
        best_alpha = alpha
        ridge.fit(X_excess, y_residual_full)
        best_co = ridge.coef_

print(f'  最佳alpha={best_alpha:.4f}, CV_MAE={best_cv_score:.4f}')

# 显示最佳co
print(f'\n--- 最佳co值 (非零) ---')
non_zero = [(fn, co) for (fn, _, _), co in zip(FLAT_ORIG, best_co) if co > 0.001]
non_zero.sort(key=lambda x: -x[1])
for fn, co in non_zero[:20]:
    old = dict((f[0], f[2]) for f in FLAT_ORIG).get(fn, 0)
    print(f'  {fn:<40} co={co:.4f} (旧={old:.4f})')

# 构建新的FLAT_FEATURES
FLAT_NEW = [(fname, bl, float(best_co[j])) for j, (fname, bl, _) in enumerate(FLAT_ORIG)]

# 计算新co的新boost
all_boosts_new_raw = np.array([compute_raw_boost(feats_list[i], best_co) for i in range(n_total)])
all_boosts_new = np.array([_dynamic_cap(b) for b in all_boosts_new_raw])
y_residual_new = y - all_boosts_new

# 用新boost重训GB
print('\n--- 重训GB (新boost) ---')

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx2, test_idx2 = next(sss2.split(X_gb, bins))

X_tr2_s = scaler_gb.fit_transform(X_gb[train_idx2])
X_te2_s = scaler_gb.transform(X_gb[test_idx2])

gb_new = GradientBoostingRegressor(n_estimators=700, max_depth=5, min_samples_leaf=3,
                                    learning_rate=0.05, subsample=0.8, random_state=42)
gb_new.fit(X_tr2_s, y[train_idx2] - all_boosts_new[train_idx2])
y_pred_new = gb_new.predict(X_te2_s) + all_boosts_new[test_idx2]
r2_new = r2_score(y[test_idx2], y_pred_new)
mae_new = mean_absolute_error(y[test_idx2], y_pred_new)
print(f'  测试集: R²={r2_new:.4f}, MAE={mae_new:.4f}')

# 全量GB
X_all2_s = scaler_gb.fit_transform(X_gb)
gb_full_new = GradientBoostingRegressor(n_estimators=700, max_depth=5, min_samples_leaf=3,
                                         learning_rate=0.05, subsample=0.8, random_state=42)
gb_full_new.fit(X_all2_s, y - all_boosts_new)

# ====== 维度占比验证 ======
cat_def = {
    '密度': ['density_dimension', 'core_peak_density_1sec_top5avg', 'core_peak_density_top5avg_1beat'],
    '平均位移': ['movement_per_second', 'burst_avg_movement', 'wide_jump_density', 'sim_pos_spread_max'],
    '配置': ['stair_density', 'stair_speed_avg', 'stair_complexity', 'stair_chord_ratio', 'trill_density',
             'jack_density', 'chord_size_entropy', 'sim_pos_spread_mean', 'multi_finger_3plus_events',
             'chord_alternation_rate', 'weighted_mf_score_per_sec', 'discrete_mf_ratio',
             'position_cluster_count', 'track_deviation_score', 'position_entropy', 'position_range_used',
             'pattern_switch_rate', 'direction_irregularity', 'hold_interference_index', 'drag_flick_ratio'],
    '耐力': ['stamina_ratio', 'tap_per_second', 'total_notes', 'tap_count', 'duration_sec',
             'rest_ratio', 'global_jack_count', 'burst_intensity_mean', 'tap_burst_top5'],
    '读谱': ['density_transition_mean', 'density_transition_std', 'tempo_change_count', 'offbeat_ratio',
             'rhythm_entropy', 'type_switch_per_sec', 'note_clutter_ratio'],
}

cat_contribs = {k: [] for k in cat_def}
for i in range(min(n_total, 300)):
    for cat_name, cat_feats in cat_def.items():
        contrib = 0
        for j, (fname, _, _) in enumerate(FLAT_ORIG):
            if fname in cat_feats and best_co[j] > 0:
                excess = compute_feature_excess(feats_list[i], fname, baselines[fname])
                contrib += best_co[j] * excess
        cat_contribs[cat_name].append(contrib)

print(f'\n维度均衡验证 (官谱前300):')
total_avg = sum(np.mean(cat_contribs[k]) for k in cat_def)
for k in ['密度', '平均位移', '配置', '耐力', '读谱']:
    avg = np.mean(cat_contribs[k])
    pct = avg / total_avg * 100 if total_avg > 0 else 0
    print(f'  {k}: {avg:.4f} ({pct:.1f}%)')

# ====== 测试谱评估 ======
print('\n--- 测试谱 (sigmoid: 沿用v7.2参) ---')

def adjust_boost_smooth(boost, gb_val, target=0.22, thresh=0.28, power=0.80):
    if boost < 2.0 or gb_val <= 0:
        return boost
    ratio = boost / gb_val
    expected = target * gb_val
    adj = expected * ((boost / expected) ** power)
    w = 1 / (1 + math.exp(-25 * (ratio - thresh)))
    return (1 - w) * boost + w * adj

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
                with open(fp, 'rb') as f: raw = f.read()
                rl = json.loads(raw.decode('utf-8')).get('META', {}).get('level')
                if rl and 5 <= (rv:=float(rl)) <= 20: rating = rv
            except: pass
        if rating is None: continue
        with open(fp, 'rb') as f: raw = f.read()
        data, _ = load_chart_from_bytes(raw)
        feats = extract_features(data)
        if not feats: continue
        chart_data.append((fn, feats, rating))
    except: continue

# 用新co+新GB预测
test_errors = []
for fn, feats, rating in chart_data:
    X = np.array([[feats.get(k, 0) for k in FN_GB]])
    Xs = scaler_gb.transform(X)
    p_gb = float(gb_full_new.predict(Xs)[0])
    p_boost = _dynamic_cap(compute_raw_boost(feats, best_co))
    p_adj = adjust_boost_smooth(p_boost, p_gb)
    pred = p_gb + p_adj
    err = pred - rating
    test_errors.append((fn, pred, rating, err, p_gb, p_boost, p_adj))

test_errors.sort(key=lambda x: x[3])
mae_test = np.mean([abs(e[3]) for e in test_errors])
pos_test = sum(1 for _,_,_,e,_,_,_ in test_errors if e > 0.01)
neg_test = sum(1 for _,_,_,e,_,_,_ in test_errors if e < -0.01)

for fn, pred, r, err, p_gb, p_boost, p_adj in test_errors:
    print(f'  {fn[:35]:<35} r={r:.1f} pred={pred:.2f} err={err:+.2f} GB={p_gb:.2f} boost={p_boost:.2f} adj={p_adj:.2f}')

print(f'\n  Ridge模型测试: MAE={mae_test:.3f}, 正偏{pos_test}/负偏{neg_test}')

# ====== 现在扫描sigmoid参数 ======
print('\n--- Sigmoid扫描 (新co+新GB) ---')

best_sig = None
best_sig_mae = float('inf')
best_sig_bal = 99

for target in [0.18, 0.20, 0.22, 0.24, 0.26]:
    for power in [0.70, 0.75, 0.80, 0.85, 0.90]:
        for thresh in [0.24, 0.26, 0.28, 0.30, 0.32]:
            errs = []
            for fn, feats, rating in chart_data:
                X = np.array([[feats.get(k, 0) for k in FN_GB]])
                Xs = scaler_gb.transform(X)
                p_gb = float(gb_full_new.predict(Xs)[0])
                p_b = _dynamic_cap(compute_raw_boost(feats, best_co))
                p_a = adjust_boost_smooth(p_b, p_gb, target=target, thresh=thresh, power=power)
                errs.append(p_gb + p_a - rating)
            m = np.mean([abs(e) for e in errs])
            p = sum(1 for e in errs if e > 0.01)
            n = sum(1 for e in errs if e < -0.01)
            b = abs(p - n)
            if b <= 6 and m < best_sig_mae:
                if b < best_sig_bal or (b == best_sig_bal and m < best_sig_mae):
                    best_sig = (target, power, thresh, b, m, p, n)
                    best_sig_mae = m
                    best_sig_bal = b
                    print(f'  target={target:.2f} power={power:.2f} thresh={thresh:.2f} MAE={m:.3f} 正偏{p}/负偏{n} 平衡={b} ***')

if best_sig:
    print(f'\n最佳sigmoid: target={best_sig[0]}, power={best_sig[1]}, thresh={best_sig[2]}')
    print(f'  MAE={best_sig[4]:.3f}, 正偏{best_sig[5]}/负偏{best_sig[6]}, 平衡={best_sig[3]}')

# ====== 保存新模型 ======
out_path = 'models/6dim_model_v7_3_ridge.pkl'
model_out = {
    'gb': gb_full_new,
    'scaler': scaler_gb,
    'feature_names': FN_GB,
    'p95_vals': P95,
    'p99_vals': P99,
    'FLAT_FEATURES': FLAT_NEW,
    'dynamic_cap': DC,
    'metrics': {'r2': r2_new, 'mae': mae_new, 'n_train': n_total, 'test_mae': mae_test},
}
os.makedirs('models', exist_ok=True)
with open(out_path, 'wb') as f:
    pickle.dump(model_out, f)
print(f'\n模型已保存: {out_path}')
print('=' * 70)
print('  v7.3 Ridge优化完成!')
print('=' * 70)
