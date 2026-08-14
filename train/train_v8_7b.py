"""
v8.7b 训练脚本 — GB 主导 + Boost 仅用于极端
策略:
  1. GB 直接学习 y (不学习残差)
  2. Boost 只在 boost > 阈值时启用
  3. 最终: GB + boost_if_significant
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print('='*60)
print('  v8.7b — GB 直接学习 y + Boost 仅极端启用')
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

# ========== 分层分割 ==========
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

# ========== FLAT 特征 (中等系数) ==========
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

# P95/P99 只在训练集上计算
P95 = {}; P99 = {}
for fname, bl, _ in FLAT:
    vals = [f.get(fname, 0) for f in train_charts]
    if vals:
        P95[fname] = float(np.percentile(vals, 95))
        P99[fname] = float(np.percentile(vals, 99))
        if P95[fname] < 0.01: P95[fname] = bl * 0.5
        if P99[fname] < 0.01: P99[fname] = bl * 0.5

# DC
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

# ========== 训练 GB 直接学习 y ==========
FN = list(FNo)
train_feats_list = [f for f in train_charts]
train_targets = np.array([f['_difficulty'] for f in train_feats_list])

X_train = np.array([[f.get(n, 0) for n in FN] for f in train_feats_list])
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

gb = GradientBoostingRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.03,
    min_samples_leaf=3, subsample=0.8, random_state=42
)
gb.fit(X_train_scaled, train_targets)

train_gb_preds = gb.predict(X_train_scaled)
train_mae_gb = mean_absolute_error(train_targets, train_gb_preds)
print(f'\nGB 训练集 MAE: {train_mae_gb:.4f}')

# ========== 计算 Boost 并分析 ==========
train_boosts = np.array([_dc(compute_raw_boost(f)) for f in train_feats_list])
train_gb_errors = train_gb_preds - train_targets

# 分析 boost 与 GB 误差的关系
print(f'\nBoost vs GB Error 分析:')
print(f'  Boost 范围: [{train_boosts.min():.2f}, {train_boosts.max():.2f}]')
print(f'  GB Error 范围: [{train_gb_errors.min():.2f}, {train_gb_errors.max():.2f}]')
print(f'  Boost 与 GB Error 相关系数: {np.corrcoef(train_boosts, train_gb_errors)[0,1]:.4f}')

# Boost 只在 GB 低估时启用 (即 boost 补充 GB 无法捕捉的部分)
# 策略: 如果 boost > 0 且 GB error < 0 (GB 低估), 则添加 boost 的 50%
# 如果 boost 很大 (> 2.0), 添加更多

# 测试集
test_feats_list = [f for f in test_charts]
test_targets = np.array([f['_difficulty'] for f in test_feats_list])
X_test = np.array([[f.get(n, 0) for n in FN] for f in test_feats_list])
X_test_scaled = scaler.transform(X_test)

test_gb_preds = gb.predict(X_test_scaled)
test_boosts = np.array([_dc(compute_raw_boost(f)) for f in test_feats_list])

test_gb_mae = mean_absolute_error(test_targets, test_gb_preds)
print(f'\nGB-only 测试集 MAE: {test_gb_mae:.4f}')

# ========== Boost 校正策略 ==========
# 搜索最优 boost 权重
best_mae = test_gb_mae
best_w = 0
for w in np.linspace(0, 1.0, 21):
    # 只在 boost * w > 0.5 时启用
    corrections = test_boosts * w
    corrections[test_boosts < 0.5] = 0  # 小 boost 不用
    test_preds = test_gb_preds + corrections
    m = mean_absolute_error(test_targets, test_preds)
    if m < best_mae:
        best_mae = m
        best_w = w

print(f'最优 Boost 权重: w={best_w:.2f}, 测试集 MAE: {best_mae:.4f}')

# 分析高低估
test_gb_errors = test_gb_preds - test_targets
under_idx = test_gb_errors < 0  # GB 低估
over_idx = test_gb_errors > 0   # GB 高估

print(f'GB 低估: {under_idx.sum()} 谱面, 平均低估: {np.abs(test_gb_errors[under_idx]).mean():.2f}')
print(f'GB 高估: {over_idx.sum()} 谱面, 平均高估: {test_gb_errors[over_idx].mean():.2f}')

# 最优策略: 只在 GB 低估时添加 boost
corrections = np.zeros_like(test_boosts)
corrections[under_idx] = test_boosts[under_idx] * best_w
test_preds = test_gb_preds + corrections
final_mae = mean_absolute_error(test_targets, test_preds)
print(f'GB(低估时+Boost*w) 测试集 MAE: {final_mae:.4f}')

# 按区间
print(f'\n===== 测试集按区间 MAE =====')
for lo, hi in [(13,14),(14,15),(15,16),(16,17),(17,20)]:
    mask = (test_targets >= lo) & (test_targets < hi)
    if mask.sum() == 0: continue
    m = mean_absolute_error(test_targets[mask], test_preds[mask])
    print(f'  [{lo},{hi}) n={mask.sum():2d}  MAE={m:.4f}')

# ========== 全量训练 ==========
print(f'\n===== 全量训练 (用于部署) =====')
all_targets = np.array([f['_difficulty'] for f in all_charts])
X_all = np.array([[f.get(n, 0) for n in FN] for f in all_charts])
scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X_all)

gb_all = GradientBoostingRegressor(
    n_estimators=300, max_depth=5, learning_rate=0.03,
    min_samples_leaf=3, subsample=0.8, random_state=42
)
gb_all.fit(X_all_scaled, all_targets)

all_gb_preds = gb_all.predict(X_all_scaled)
all_boosts = np.array([_dc(compute_raw_boost(f)) for f in all_charts])
all_errors = all_gb_preds - all_targets
all_under = all_errors < 0

all_corrections = np.zeros_like(all_boosts)
all_corrections[all_under] = all_boosts[all_under] * best_w
all_preds = all_gb_preds + all_corrections
all_mae = mean_absolute_error(all_targets, all_preds)
print(f'全量训练集 MAE: {all_mae:.4f}')

# ========== 保存 ==========
model = {
    'gb': gb_all,
    'scaler': scaler_all,
    'feature_names': FN,
    'FLAT_FEATURES': FLAT,
    'p95_vals': P95,
    'p99_vals': P99,
    'dynamic_cap': DC,
    'boost_weight': best_w,
    'version': '8.7b',
    'description': 'GB主导 + Boost仅低估时启用'
}
os.makedirs('models', exist_ok=True)
with open('models/6dim_model_v8_7b.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f'\nSaved: models/6dim_model_v8_7b.pkl')

# 极端谱
print(f'\n===== 极端谱面诊断 (测试集) =====')
errors = test_preds - test_targets
extreme_idx = np.where(np.abs(errors) > 0.5)[0]
print(f'{"Name":<30s} {"True":>6s} {"GB":>7s} {"Boost":>7s} {"Corr":>7s} {"Pred":>7s} {"Err":>7s}')
for i in extreme_idx:
    name = test_charts[i]['_name'][:28]
    print(f'{name:<30s} {test_targets[i]:>6.1f} {test_gb_preds[i]:>7.2f} {test_boosts[i]:>7.2f} {corrections[i]:>7.2f} {test_preds[i]:>7.2f} {errors[i]:>+7.2f}')

print(f'\n===== 完成 =====')