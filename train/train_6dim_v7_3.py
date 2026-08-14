"""
v7.3 权重优化 v2 — 仅IN/AT谱做Ridge + 迭代重训

关键改进:
  1. 只对IN/AT官谱做Ridge (difficulty>12), 这是模型真正需要校准的范围
  2. 迭代3轮: 学习co → 重训GB → 重新计算残差 → 再学co
  3. 最后在IN/AT测试集上扫描sigmoi参数
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, json, os, pickle, numpy as np, math, re
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, '.')
from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from unified_parser import load_chart_from_bytes

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('models/6dim_model_v7_2.pkl', 'rb') as f:
    v7m = pickle.load(f)
P95 = v7m['p95_vals']; P99 = v7m['p99_vals']
FLAT_ORIG = v7m['FLAT_FEATURES']; FN_GB = v7m['feature_names']
feat_names = [f[0] for f in FLAT_ORIG]
n_feats = len(feat_names)

print('=' * 70)
print('  v7.3 — IN/AT限定Ridge + 迭代重训')
print('=' * 70)

# ====== 加载官谱 ======
CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)

all_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    for lv in ['IN','AT']:  # 只取IN/AT
        if lv in info['levels'] and lv in diffs:
            all_items.append({'folder':fn,'filepath':info['levels'][lv],'difficulty':diffs[lv],'level':lv})
print(f'IN/AT官谱: {len(all_items)}')

feats_list, labels, names_list = [], [], []
for item in all_items:
    try:
        cd = load_chart_json(item['filepath'])
        feats = extract_features(cd)
        if feats:
            feats_list.append(feats)
            labels.append(item['difficulty'])
            names_list.append(item['folder'])
    except: pass

n_all = len(feats_list)
labels = np.array(labels)
print(f'  提取: {n_all}, 难度 {labels.min():.1f}~{labels.max():.1f}')
print(f'  分布: <13:{sum(labels<13)} 13-15:{sum((labels>=13)&(labels<15))} 15-16:{sum((labels>=15)&(labels<16))} 16+:{sum(labels>=16)}')

# ====== Excess矩阵 ======
def compute_excess(feats, fname, bl):
    val = feats.get(fname, 0)
    pv = P95.get(fname, 0)
    thresh = max(pv * 0.55, bl * 0.5)
    if val <= thresh: return 0.0
    excess = (val / thresh - 1.0) ** 0.70
    if val > max(P99.get(fname, 0), bl * 0.5):
        p99_excess = (val / max(P99.get(fname, 0), bl * 0.5) - 1.0)
        excess += 0.5 * max(0, p99_excess) ** 0.70
    return excess

X_excess = np.zeros((n_all, n_feats))
for i in range(n_all):
    for j, (fname, bl, _) in enumerate(FLAT_ORIG):
        X_excess[i, j] = compute_excess(feats_list[i], fname, bl)

X_gb = np.array([[f.get(n,0) for n in FN_GB] for f in feats_list])
y = labels.copy()

DC = {'knee': 1.0, 'power': 0.90}
def _dynamic_cap(raw):
    return raw if raw <= DC['knee'] else DC['knee'] + (raw - DC['knee']) ** DC['power']

def compute_raw_boost(feats, co_arr):
    raw = 0.0
    for j, (fname, bl, _) in enumerate(FLAT_ORIG):
        raw += co_arr[j] * compute_excess(feats, fname, bl)
    return raw

def adjust_boost_smooth(boost, gb_val, target=0.22, thresh=0.28, power=0.80):
    if boost < 2.0 or gb_val <= 0: return boost
    ratio = boost / gb_val
    expected = target * gb_val
    adj = expected * ((boost / expected) ** power)
    w = 1 / (1 + math.exp(-25 * (ratio - thresh)))
    return (1 - w) * boost + w * adj

# 划分训练/测试(IN/AT)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
bins = np.digitize(y, bins=[12,13,14,15,16,17,18])
train_idx, test_idx = next(sss.split(X_gb, bins))
y_test = y[test_idx]

# ====== 迭代优化 ======
N_ITER = 3
co_best = None
gb_best = None
scaler_best = None
mae_best = float('inf')

# 初始co: 使用v7.2的co但全乘0.5作为起点(因为BPM修正后excess变大了)
co_current = np.array([f[2] for f in FLAT_ORIG]) * 0.6

for it in range(N_ITER):
    print(f'\n--- 迭代 {it+1}/{N_ITER} ---')
    
    # 计算当前boost
    all_boosts = np.array([_dynamic_cap(compute_raw_boost(feats_list[i], co_current)) for i in range(n_all)])
    boost_range = f'{all_boosts.min():.2f}~{all_boosts.max():.2f}'
    
    # 重训GB
    X_tr = X_gb[train_idx]; y_tr = y[train_idx]
    X_te = X_gb[test_idx]; y_te = y[test_idx]
    boosts_tr = all_boosts[train_idx]; boosts_te = all_boosts[test_idx]
    
    scaler = StandardScaler()
    gb_m = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                      learning_rate=0.05, subsample=0.8, random_state=42)
    gb_m.fit(scaler.fit_transform(X_tr), y_tr - boosts_tr)
    preds_te = gb_m.predict(scaler.transform(X_te)) + boosts_te
    r2 = r2_score(y_te, preds_te)
    mae = mean_absolute_error(y_te, preds_te)
    print(f'  GB: R²={r2:.4f}, MAE={mae:.4f}, Boost范围={boost_range}')
    
    # 全量GB
    scaler_all = StandardScaler()
    gb_full = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                         learning_rate=0.05, subsample=0.8, random_state=42)
    X_all_s = scaler_all.fit_transform(X_gb)
    gb_full.fit(X_all_s, y - all_boosts)
    
    # 计算残差: GB_error = y - GB_predict (不包括boost)
    y_gb_pred = gb_full.predict(X_all_s)
    y_residual = y - y_gb_pred
    
    # Ridge拟合 co * excess = y_residual
    # Alpha从宽范围中CV选择
    from sklearn.model_selection import cross_val_score
    best_alpha_ridge = 1.0
    best_cv = float('inf')
    for alpha in [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]:
        ridge = Ridge(alpha=alpha, fit_intercept=False, positive=True)
        scores = -cross_val_score(ridge, X_excess, y_residual, cv=5, scoring='neg_mean_absolute_error')
        if scores.mean() < best_cv:
            best_cv = scores.mean()
            best_alpha_ridge = alpha
    
    ridge = Ridge(alpha=best_alpha_ridge, fit_intercept=False, positive=True)
    ridge.fit(X_excess, y_residual)
    co_new = ridge.coef_
    
    print(f'  Ridge alpha={best_alpha_ridge:.3f}, CV_MAE={best_cv:.4f}')
    
    # 显示维度变化
    cat_def = {
        '密度': ['density_dimension', 'core_peak_density_1sec_top5avg', 'core_peak_density_top5avg_1beat'],
        '位移': ['movement_per_second', 'burst_avg_movement', 'wide_jump_density', 'sim_pos_spread_max'],
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
    cat_sum_new = {}
    for cat_name, cat_feats in cat_def.items():
        s = 0
        for j, (fname, _, _) in enumerate(FLAT_ORIG):
            if fname in cat_feats:
                s += co_new[j]
        cat_sum_new[cat_name] = s
    total_new = sum(cat_sum_new.values())
    dim_str = ' '.join(f'{k}={v:.3f}({v/total_new*100:.0f}%)' for k,v in cat_sum_new.items())
    print(f'  维度co: {dim_str}')
    
    # 混合: co = 0.3*old + 0.7*new (平滑过渡)
    co_current = 0.3 * co_current + 0.7 * co_new
    
    # 检查是否改善
    if mae < mae_best:
        mae_best = mae
        co_best = co_current.copy()
        gb_best = gb_full
        scaler_best = scaler_all

# ====== 用最优co + 最优GB做最终评估 ======
print('\n' + '=' * 70)
print('最终模型评估')

# 全量GB重训
all_boosts_final = np.array([_dynamic_cap(compute_raw_boost(feats_list[i], co_best)) for i in range(n_all)])
scaler_final = StandardScaler()
gb_final = GradientBoostingRegressor(n_estimators=700, max_depth=5, min_samples_leaf=3,
                                      learning_rate=0.05, subsample=0.8, random_state=42)
gb_final.fit(scaler_final.fit_transform(X_gb), y - all_boosts_final)

# 测试谱评估
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

print(f'\n测试谱: {len(chart_data)}个')

# Sigmoid扫描
print(f'\nSigmoid扫描:')
best_sig = None
for target in [0.16, 0.18, 0.20, 0.22, 0.24]:
    for power in [0.70, 0.75, 0.78, 0.80, 0.83, 0.85, 0.88]:
        for thresh in [0.24, 0.26, 0.28, 0.30, 0.32, 0.34]:
            errs = []
            for fn, feats, rating in chart_data:
                X = np.array([[feats.get(k,0) for k in FN_GB]])
                Xs = scaler_final.transform(X)
                p_gb = float(gb_final.predict(Xs)[0])
                p_b = _dynamic_cap(compute_raw_boost(feats, co_best))
                p_a = adjust_boost_smooth(p_b, p_gb, target=target, thresh=thresh, power=power)
                errs.append(p_gb + p_a - rating)
            m = np.mean([abs(e) for e in errs])
            p = sum(1 for e in errs if e > 0.01)
            n = sum(1 for e in errs if e < -0.01)
            b = abs(p - n)
            if m < 0.45 and b <= 5:
                mark = ' ***' if b <= 2 else ' *'
                print(f'  t={target:.2f} p={power:.2f} th={thresh:.2f} MAE={m:.3f} 正{p}/负{n} 平衡={b}{mark}')
                if best_sig is None or b < best_sig[3] or (b==best_sig[3] and m<best_sig[4]):
                    best_sig = (target, power, thresh, b, m, p, n)

if best_sig:
    print(f'\n=== 最优sigmoid: target={best_sig[0]}, power={best_sig[1]}, thresh={best_sig[2]} ===')
    print(f'  MAE={best_sig[4]:.3f}, 正偏{best_sig[5]}/负偏{best_sig[6]}')
    
    # 详细结果
    print(f'\n详细:')
    errs_final = []
    for fn, feats, rating in chart_data:
        X = np.array([[feats.get(k,0) for k in FN_GB]])
        Xs = scaler_final.transform(X)
        p_gb = float(gb_final.predict(Xs)[0])
        p_b = _dynamic_cap(compute_raw_boost(feats, co_best))
        p_a = adjust_boost_smooth(p_b, p_gb, target=best_sig[0], thresh=best_sig[2], power=best_sig[1])
        pred = p_gb + p_a
        errs_final.append((fn, pred, rating, pred-rating, p_gb, p_b, p_a))
    errs_final.sort(key=lambda x: x[3])
    
    for fn, pred, r, err, p_gb, p_b, p_a in errs_final:
        print(f'  {fn[:38]:<38} r={r:.1f} pred={pred:.2f} err={err:+.2f} GB={p_gb:.2f} boost={p_b:.2f} adj={p_a:.2f}')
    
    final_mae = np.mean([abs(e[3]) for e in errs_final])
    pos_f = sum(1 for _,_,_,e,_,_,_ in errs_final if e > 0.01)
    neg_f = sum(1 for _,_,_,e,_,_,_ in errs_final if e < -0.01)
    print(f'\n  最终MAE={final_mae:.3f}, 正偏{pos_f}/负偏{neg_f}')

# ====== 保存模型 ======
FLAT_FINAL = [(fname, bl, float(co_best[j])) for j, (fname, bl, _) in enumerate(FLAT_ORIG)]

cat_sum = {}
for j, (fname, _, _) in enumerate(FLAT_ORIG):
    for cat_name, cat_feats in {
        '密度': ['density_dimension','core_peak_density_1sec_top5avg','core_peak_density_top5avg_1beat'],
        '位移': ['movement_per_second','burst_avg_movement','wide_jump_density','sim_pos_spread_max'],
        '配置': ['stair_density','stair_speed_avg','stair_complexity','stair_chord_ratio','trill_density',
                 'jack_density','chord_size_entropy','sim_pos_spread_mean','multi_finger_3plus_events',
                 'chord_alternation_rate','weighted_mf_score_per_sec','discrete_mf_ratio',
                 'position_cluster_count','track_deviation_score','position_entropy','position_range_used',
                 'pattern_switch_rate','direction_irregularity','hold_interference_index','drag_flick_ratio'],
        '耐力': ['stamina_ratio','tap_per_second','total_notes','tap_count','duration_sec',
                 'rest_ratio','global_jack_count','burst_intensity_mean','tap_burst_top5'],
        '读谱': ['density_transition_mean','density_transition_std','tempo_change_count','offbeat_ratio',
                 'rhythm_entropy','type_switch_per_sec','note_clutter_ratio'],
    }.items():
        if fname in cat_feats:
            cat_sum[cat_name] = cat_sum.get(cat_name, 0) + co_best[j]

total_cat = sum(cat_sum.values())
print(f'\n维度co占比:')
for k,v in cat_sum.items():
    print(f'  {k}: {v:.3f} ({v/total_cat*100:.1f}%)')

out_path = 'models/6dim_model_v7_3.pkl'
model_out = {
    'gb': gb_final, 'scaler': scaler_final,
    'feature_names': FN_GB,
    'p95_vals': P95, 'p99_vals': P99,
    'FLAT_FEATURES': FLAT_FINAL,
    'dynamic_cap': DC,
    'metrics': {'mae': final_mae if 'final_mae' in dir() else 0, 'n_train': n_all},
}
if best_sig:
    model_out['sigmoid_params'] = {'target': best_sig[0], 'power': best_sig[1], 'thresh': best_sig[2]}
os.makedirs('models', exist_ok=True)
with open(out_path, 'wb') as f:
    pickle.dump(model_out, f)
print(f'\n模型已保存: {out_path}')
