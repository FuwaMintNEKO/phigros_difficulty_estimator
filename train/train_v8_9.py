"""
v8.9 — GB 主导 + Boost 作为特征
策略:
  1. GB 直接学习 y, 但 boost 作为额外特征加入
  2. 这样 GB 可以学习什么时候用 boost, 什么时候不用
  3. 超参数网格搜索
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os, pickle, numpy as np, math, json
sys.path.insert(0, '.')
from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print('='*60)
print('  v8.9 — GB 主导 + Boost 作为特征 + 网格搜索')
print('='*60)

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)

all_charts = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    for lv in ['IN', 'AT']:
        if lv not in info.get('levels', {}): continue
        if lv not in song_difficulties[sid]: continue
        try:
            cd = load_chart_json(info['levels'][lv])
            feats = extract_features(cd)
            if feats:
                feats['_difficulty'] = song_difficulties[sid][lv]
                feats['_name'] = fn[:30]
                all_charts.append(feats)
        except Exception as e: pass

exclude_patterns = ['snowmelt', 'snowdance', 'snow dance']
all_charts = [f for f in all_charts if not any(p.lower() in f['_name'].lower() for p in exclude_patterns)]
print(f'总谱面数: {len(all_charts)}')

# 分层分割
diffs = np.array([f['_difficulty'] for f in all_charts])
bins = np.digitize(diffs, bins=[13, 14, 15, 16, 17])
train_mask = np.zeros(len(all_charts), dtype=bool)
test_mask = np.zeros(len(all_charts), dtype=bool)

np.random.seed(42)
for b in range(1, 6):
    idx = np.where(bins == b)[0]
    if len(idx) == 0: continue
    tr_idx, te_idx = train_test_split(idx, test_size=0.25, random_state=42)
    train_mask[tr_idx] = True
    test_mask[te_idx] = True

train_charts = [all_charts[i] for i in range(len(all_charts)) if train_mask[i]]
test_charts = [all_charts[i] for i in range(len(all_charts)) if test_mask[i]]
print(f'训练集: {len(train_charts)} 谱面, 测试集: {len(test_charts)} 谱面')

# FLAT 特征 (用于 Boost)
FLAT = [
    ('density_dimension',              1.0,    0.50),
    ('stair_rate_per_sec',             2.0,    0.25),
    ('stair_complexity',               0.2,    0.08),
    ('chord_size_entropy',             0.5,    0.08),
    ('chord_alternation_rate',         0.5,    0.40),
    ('weighted_mf_score_per_sec',      10.0,   0.25),
    ('position_entropy',               2.0,    0.08),
    ('avg_chord_size_poly',            2.0,    0.12),
    ('position_range_used',            0.5,    0.08),
    ('trill_density',                  2.0,    0.12),
    ('multi_finger_3plus_events',      0.5,    0.08),
    ('above_avg_density_mean',         4.0,    1.00),
    ('above_avg_duration_sec',         30.0,   0.20),
    ('total_notes',                    400.0,  0.60),
    ('tap_burst_top5',                 0.5,    0.15),
    ('tempo_change_count',             50.0,   0.08),
    ('type_switch_per_sec',            0.4,    0.20),
    ('density_transition_std',         0.2,    0.15),
    ('density_transition_mean',        0.1,    0.08),
    ('note_clutter_ratio',             0.1,    0.15),
    ('rhythm_entropy',                 2.5,    0.12),
    ('hold_interference_index',        0.3,    0.15),
    ('jline_movement_density',         50.0,   0.20),
    ('jline_rotate_density',           20.0,   0.12),
    ('jline_disappear_density',        20.0,   0.12),
    ('above_below_cross',              0.3,    0.12),
    ('fast_note_density_16th',         4.0,    0.30),
    ('fast_note_density_32nd',         2.0,    0.60),
    ('fast_note_density_24th',         1.0,    0.40),
    ('fast_note_density_48th',         0.5,    0.50),
    ('fast_note_density_64th',         0.3,    0.40),
    ('rhythm_type_count',              3.0,    0.40),
]

P95 = {}; P99 = {}
for fname, bl, _ in FLAT:
    vals = [f.get(fname, 0) for f in train_charts]
    if vals:
        P95[fname] = float(np.percentile(vals, 95))
        P99[fname] = float(np.percentile(vals, 99))
        if P95[fname] < 0.01: P95[fname] = bl * 0.5
        if P99[fname] < 0.01: P99[fname] = bl * 0.5

with open('models/6dim_model_v8_4.pkl', 'rb') as f:
    v84 = pickle.load(f)
DC = v84['dynamic_cap']
FNo = v84['feature_names']

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

def compute_raw_boost(feats):
    raw = 0.0
    for fname, bl, co in FLAT:
        raw += co * compute_excess(feats, fname, bl)
    return raw

def _dc(r):
    if r <= DC['knee']: return r
    return DC['knee'] + (r - DC['knee']) ** DC['power']

# ========== 构建特征矩阵 (加入 boost) ==========
FN = list(FNo)
train_feats_list = [f for f in train_charts]
train_targets = np.array([f['_difficulty'] for f in train_feats_list])

# 计算 boost 并作为特征加入
train_boosts = np.array([_dc(compute_raw_boost(f)) for f in train_feats_list])

X_train_base = np.array([[f.get(n, 0) for n in FN] for f in train_feats_list])
X_train = np.column_stack([X_train_base, train_boosts])  # 加入 boost 作为第 171 个特征
FN_extended = FN + ['__boost__']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 测试集
test_feats_list = [f for f in test_charts]
test_targets = np.array([f['_difficulty'] for f in test_feats_list])
test_boosts = np.array([_dc(compute_raw_boost(f)) for f in test_feats_list])

X_test_base = np.array([[f.get(n, 0) for n in FN] for f in test_feats_list])
X_test = np.column_stack([X_test_base, test_boosts])
X_test_scaled = scaler.transform(X_test)

# ========== 网格搜索 ==========
print(f'\n===== 网格搜索 =====')
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.03, 0.05],
    'min_samples_leaf': [3, 5],
    'subsample': [0.8, 1.0],
}
gb = GradientBoostingRegressor(random_state=42)
grid = GridSearchCV(gb, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=1, verbose=1)
grid.fit(X_train_scaled, train_targets)

print(f'\n最优参数: {grid.best_params_}')
print(f'CV MAE: {-grid.best_score_:.4f}')

# 测试集评估
best_gb = grid.best_estimator_
test_preds = best_gb.predict(X_test_scaled)
test_mae = mean_absolute_error(test_targets, test_preds)
print(f'测试集 MAE: {test_mae:.4f}')

# 特征重要性
importances = best_gb.feature_importances_
boost_importance = importances[-1]  # boost 特征的重要性
print(f'Boost 特征重要性: {boost_importance:.4f} (排名 {np.argsort(-importances)[0]+1}/{len(importances)})')

# 按区间
print(f'\n===== 测试集按区间 MAE =====')
for lo, hi in [(13,14),(14,15),(15,16),(16,17),(17,20)]:
    mask = (test_targets >= lo) & (test_targets < hi)
    if mask.sum() == 0: continue
    m = mean_absolute_error(test_targets[mask], test_preds[mask])
    print(f'  [{lo},{hi}) n={mask.sum():2d}  MAE={m:.4f}')

# ========== 全量训练 ==========
print(f'\n===== 全量训练 =====')
all_targets = np.array([f['_difficulty'] for f in all_charts])
all_boosts = np.array([_dc(compute_raw_boost(f)) for f in all_charts])
X_all_base = np.array([[f.get(n, 0) for n in FN] for f in all_charts])
X_all = np.column_stack([X_all_base, all_boosts])

scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X_all)

gb_all = GradientBoostingRegressor(**grid.best_params_, random_state=42)
gb_all.fit(X_all_scaled, all_targets)

all_preds = gb_all.predict(X_all_scaled)
all_mae = mean_absolute_error(all_targets, all_preds)
print(f'全量 MAE: {all_mae:.4f}')

# ========== 保存 ==========
model = {
    'gb': gb_all,
    'scaler': scaler_all,
    'feature_names': FN_extended,
    'FLAT_FEATURES': FLAT,
    'p95_vals': P95,
    'p99_vals': P99,
    'dynamic_cap': DC,
    'version': '8.9',
    'description': 'GB主导 + Boost作为特征 + 网格搜索'
}
os.makedirs('models', exist_ok=True)
with open('models/6dim_model_v8_9.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f'\nSaved: models/6dim_model_v8_9.pkl')

# 极端谱
print(f'\n===== 极端谱面诊断 (测试集) =====')
errors = test_preds - test_targets
extreme_idx = np.where(np.abs(errors) > 0.5)[0]
print(f'{"Name":<30s} {"True":>6s} {"Pred":>7s} {"Err":>7s}')
for i in extreme_idx:
    name = test_charts[i]['_name'][:28]
    print(f'{name:<30s} {test_targets[i]:>6.1f} {test_preds[i]:>7.2f} {errors[i]:>+7.2f}')

print(f'\n===== 完成 =====')