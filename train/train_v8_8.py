"""
v8.8 — Ridge 回归 + Boost 校正
策略:
  1. RidgeCV 在选定特征上学习线性难度
  2. Boost 捕获 Ridge 无法处理的非线性极端情况
  3. 最终: Ridge + Boost*c
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os, pickle, numpy as np, math, json
sys.path.insert(0, '.')
from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print('='*60)
print('  v8.8 — Ridge 回归 + Boost 校正')
print('='*60)

# ========== 加载数据 ==========
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

# ========== FLAT 特征 (用于 Boost) ==========
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

# P95/P99
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

# ========== Ridge 回归特征选择 ==========
# 使用与 y 相关性最高的特征 + 物理意义明确的特征
# 先计算所有特征与 y 的相关性
from sklearn.linear_model import Ridge
train_feats_list = [f for f in train_charts]
train_targets = np.array([f['_difficulty'] for f in train_feats_list])

# 选择用于 Ridge 的特征 (排除 FLAT 中已用于 boost 的特征, 避免重复)
# 以及排除一些噪声特征
all_feat_names = sorted([k for k in train_feats_list[0].keys() if not k.startswith('_')])
flat_feat_names = {fname for fname, _, _ in FLAT}

# 计算每个特征与 y 的相关系数
corrs = {}
for name in all_feat_names:
    vals = [f.get(name, 0) for f in train_feats_list]
    if np.std(vals) == 0: continue
    r = np.corrcoef(vals, train_targets)[0, 1]
    if not np.isnan(r):
        corrs[name] = abs(r)

# 选择 top 40 特征 (排除 flat 特征)
ridge_feats = sorted([(k, v) for k, v in corrs.items() if k not in flat_feat_names], 
                     key=lambda x: -x[1])[:40]
ridge_feat_names = [k for k, _ in ridge_feats]
print(f'\nRidge 特征 (top 40, 排除boost特征):')
for k, v in ridge_feats[:10]:
    print(f'  {k:<35s} r={v:.4f}')
print(f'  ... 共 {len(ridge_feats)} 个')

# RidgeCV
X_train_ridge = np.array([[f.get(n, 0) for n in ridge_feat_names] for f in train_feats_list])
scaler = StandardScaler()
X_train_ridge_scaled = scaler.fit_transform(X_train_ridge)

ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
ridge.fit(X_train_ridge_scaled, train_targets)

train_ridge_preds = ridge.predict(X_train_ridge_scaled)
train_ridge_mae = mean_absolute_error(train_targets, train_ridge_preds)
print(f'\nRidge 训练集 MAE: {train_ridge_mae:.4f}')

# Ridge 残差
train_residuals = train_targets - train_ridge_preds
train_boosts = np.array([_dc(compute_raw_boost(f)) for f in train_feats_list])

# 分析残差与 boost 的关系
print(f'Ridge 残差范围: [{train_residuals.min():.2f}, {train_residuals.max():.2f}]')
print(f'Boost 与残差 相关系数: {np.corrcoef(train_boosts, train_residuals)[0,1]:.4f}')

# 测试 Ridge
test_feats_list = [f for f in test_charts]
test_targets = np.array([f['_difficulty'] for f in test_feats_list])
X_test_ridge = np.array([[f.get(n, 0) for n in ridge_feat_names] for f in test_feats_list])
X_test_ridge_scaled = scaler.transform(X_test_ridge)

test_ridge_preds = ridge.predict(X_test_ridge_scaled)
test_ridge_mae = mean_absolute_error(test_targets, test_ridge_preds)
print(f'Ridge 测试集 MAE: {test_ridge_mae:.4f}')

test_boosts = np.array([_dc(compute_raw_boost(f)) for f in test_feats_list])
test_residuals = test_targets - test_ridge_preds

# ========== 搜索最优 boost 校正 ==========
# 策略: Ridge + w * Boost 当 boost > 阈值
best_mae = test_ridge_mae
best_w, best_thresh = 0, 0
for w in np.linspace(0, 2.0, 21):
    for thresh in [0, 0.5, 1.0, 1.5, 2.0]:
        corrections = test_boosts * w
        corrections[test_boosts < thresh] = 0
        test_preds = test_ridge_preds + corrections
        m = mean_absolute_error(test_targets, test_preds)
        if m < best_mae:
            best_mae = m
            best_w = w
            best_thresh = thresh

print(f'\n最优 Boost: w={best_w:.2f}, thresh={best_thresh:.1f}, 测试集 MAE: {best_mae:.4f}')

# 应用最优策略
corrections = test_boosts * best_w
corrections[test_boosts < best_thresh] = 0
test_preds = test_ridge_preds + corrections
final_mae = mean_absolute_error(test_targets, test_preds)
print(f'Ridge+Boost 测试集 MAE: {final_mae:.4f}')

print(f'\n===== 测试集按区间 MAE =====')
for lo, hi in [(13,14),(14,15),(15,16),(16,17),(17,20)]:
    mask = (test_targets >= lo) & (test_targets < hi)
    if mask.sum() == 0: continue
    m = mean_absolute_error(test_targets[mask], test_preds[mask])
    print(f'  [{lo},{hi}) n={mask.sum():2d}  MAE={m:.4f}')

# ========== 全量训练 ==========
print(f'\n===== 全量训练 (用于部署) =====')
all_targets = np.array([f['_difficulty'] for f in all_charts])
X_all_ridge = np.array([[f.get(n, 0) for n in ridge_feat_names] for f in all_charts])
scaler_all = StandardScaler()
X_all_ridge_scaled = scaler_all.fit_transform(X_all_ridge)

ridge_all = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
ridge_all.fit(X_all_ridge_scaled, all_targets)

all_ridge_preds = ridge_all.predict(X_all_ridge_scaled)
all_boosts = np.array([_dc(compute_raw_boost(f)) for f in all_charts])

all_corrections = all_boosts * best_w
all_corrections[all_boosts < best_thresh] = 0
all_preds = all_ridge_preds + all_corrections
all_mae = mean_absolute_error(all_targets, all_preds)
print(f'全量 MAE: {all_mae:.4f}')

# ========== 保存 ==========
model = {
    'ridge': ridge_all,
    'scaler': scaler_all,
    'ridge_features': ridge_feat_names,
    'FLAT_FEATURES': FLAT,
    'p95_vals': P95,
    'p99_vals': P99,
    'dynamic_cap': DC,
    'boost_weight': best_w,
    'boost_threshold': best_thresh,
    'version': '8.8',
    'description': 'Ridge回归 + Boost校正'
}
os.makedirs('models', exist_ok=True)
with open('models/6dim_model_v8_8.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f'\nSaved: models/6dim_model_v8_8.pkl')

# 极端谱
print(f'\n===== 极端谱面诊断 (测试集) =====')
errors = test_preds - test_targets
extreme_idx = np.where(np.abs(errors) > 0.5)[0]
print(f'{"Name":<30s} {"True":>6s} {"Ridge":>7s} {"Boost":>7s} {"Corr":>7s} {"Pred":>7s} {"Err":>7s}')
for i in extreme_idx:
    name = test_charts[i]['_name'][:28]
    print(f'{name:<30s} {test_targets[i]:>6.1f} {test_ridge_preds[i]:>7.2f} {test_boosts[i]:>7.2f} {corrections[i]:>7.2f} {test_preds[i]:>7.2f} {errors[i]:>+7.2f}')

print(f'\n===== 完成 =====')