"""
v8.7 训练脚本 — 正确分割 + Boost 适量化 + GB 正则化
问题诊断:
  - v8.6 boost范围 [0.22, 2.76] 太小，GB拟合全部难度 → 过拟合
  - 需要boost主导预测(50-80%)，GB仅做修正
修复:
  1. 80/20 分层分割，P95/P99只在训练集上算
  2. Boost系数放大，使boost覆盖大部分难度范围
  3. GB减少到100棵树，加正则化
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
import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print('='*60)
print('  v8.7 — 正确分割 + Boost 主导 + GB 正则化')
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

# 排除测试谱
exclude_patterns = ['snowmelt', 'snowdance', 'snow dance']
all_charts = [f for f in all_charts if not any(p.lower() in f['_name'].lower() for p in exclude_patterns)]

print(f'总谱面数: {len(all_charts)}')

# ========== 1. 分层分割 (80/20) ==========
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

# ========== 2. FLAT 特征定义 (v8.6 基础上, 增大系数) ==========
# 为使boost覆盖50-80%难度范围，系数需放大约4-5倍
# 密度维度: 原来0.12 → 现在0.50
# 配置维度: 原来0.38 → 现在1.50
# 等等

FLAT = [
    # ========== 密度 (1特征) ==========
    ('density_dimension',              1.0,    0.50),

    # ========== 配置 (10特征) ==========
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

    # ========== 耐力 (4特征) ==========
    ('above_avg_density_mean',         4.0,    1.00),
    ('above_avg_duration_sec',         30.0,   0.20),
    ('total_notes',                    400.0,  0.60),
    ('tap_burst_top5',                 0.5,    0.15),

    # ========== 读谱 (11特征) ==========
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

    # ========== 高速 (6特征) ==========
    ('fast_note_density_16th',         4.0,    0.30),
    ('fast_note_density_32nd',         2.0,    0.60),
    ('fast_note_density_24th',         1.0,    0.40),
    ('fast_note_density_48th',         0.5,    0.50),
    ('fast_note_density_64th',         0.3,    0.40),
    ('rhythm_type_count',              3.0,    0.40),
]

# 维度 co 总和
density_co = sum(c for n,_,c in FLAT if n == 'density_dimension')
config_co = sum(c for n,_,c in FLAT if n in ['stair_rate_per_sec','stair_complexity','chord_size_entropy','chord_alternation_rate','weighted_mf_score_per_sec','position_entropy','avg_chord_size_poly','position_range_used','trill_density','multi_finger_3plus_events'])
stamina_co = sum(c for n,_,c in FLAT if n in ['above_avg_density_mean','above_avg_duration_sec','total_notes','tap_burst_top5'])
reading_co = sum(c for n,_,c in FLAT if n in ['tempo_change_count','type_switch_per_sec','density_transition_std','density_transition_mean','note_clutter_ratio','rhythm_entropy','hold_interference_index','jline_movement_density','jline_rotate_density','jline_disappear_density','above_below_cross'])
fast_co = sum(c for n,_,c in FLAT if n.startswith('fast_') or n == 'rhythm_type_count')

print(f'\n维度 co 总和 (放大后):')
print(f'  密度: {density_co:.2f} (1特征)')
print(f'  配置: {config_co:.2f} (10特征, 平均 co={config_co/10:.3f})')
print(f'  耐力: {stamina_co:.2f} (4特征, 平均 co={stamina_co/4:.3f})')
print(f'  读谱: {reading_co:.2f} (11特征, 平均 co={reading_co/11:.3f})')
print(f'  高速: {fast_co:.2f} (6特征)')
print(f'  总计: {sum(c for _,_,c in FLAT):.2f} ({len(FLAT)}特征)')

# ========== 3. P95/P99 只在训练集上计算 ==========
P95 = {}; P99 = {}
for fname, bl, _ in FLAT:
    vals = [f.get(fname, 0) for f in train_charts]
    if vals:
        P95[fname] = float(np.percentile(vals, 95))
        P99[fname] = float(np.percentile(vals, 99))
        if P95[fname] < 0.01: P95[fname] = bl * 0.5
        if P99[fname] < 0.01: P99[fname] = bl * 0.5

# ========== 4. Dynamic Cap (从v8.4继承) ==========
with open('models/6dim_model_v8_4.pkl', 'rb') as f:
    v84 = pickle.load(f)
DC = v84['dynamic_cap']
FNo = v84['feature_names']

# ========== 5. Boost 计算函数 ==========
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

def adjust_boost_smooth(boost, gb_val):
    if boost < 2.0: return boost
    ratio = boost / gb_val if gb_val > 0 else 0
    expected = 0.32 * gb_val
    if expected <= 0 or boost <= 0: return boost
    adj = expected * ((boost / expected) ** 0.75)
    w = 1 / (1 + math.exp(-25 * (ratio - 0.22)))
    return (1 - w) * boost + w * adj

# ========== 6. 计算训练集 boost 和 GB 目标 ==========
train_feats_list = [f for f in train_charts]
train_raw_boosts = np.array([_dc(compute_raw_boost(f)) for f in train_feats_list])
train_targets = np.array([f['_difficulty'] for f in train_feats_list])

print(f'\n训练集 boost 范围: [{train_raw_boosts.min():.2f}, {train_raw_boosts.max():.2f}]')
print(f'训练集 目标 范围: [{train_targets.min():.2f}, {train_targets.max():.2f}]')
print(f'Boost/目标比值: {train_raw_boosts.mean()/train_targets.mean():.2%}')

# GB 目标是 y - boost
gb_targets = train_targets - train_raw_boosts
print(f'GB 残差范围: [{gb_targets.min():.2f}, {gb_targets.max():.2f}]')

# ========== 7. 训练 GB (减少复杂度) ==========
FN = list(FNo)
X_train = np.array([[f.get(n, 0) for n in FN] for f in train_feats_list])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

gb = GradientBoostingRegressor(
    n_estimators=200,          # 增加树数
    max_depth=4,               # 更深, 捕获更多交互
    learning_rate=0.05,
    min_samples_leaf=3,        # 减少正则化
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_scaled, gb_targets)

train_preds = gb.predict(X_train_scaled) + train_raw_boosts
train_mae = mean_absolute_error(train_targets, train_preds)
print(f'\n训练集 MAE: {train_mae:.4f}')

# ========== 8. 测试集评估 ==========
test_feats_list = [f for f in test_charts]
test_raw_boosts = np.array([_dc(compute_raw_boost(f)) for f in test_feats_list])
test_targets = np.array([f['_difficulty'] for f in test_feats_list])

X_test = np.array([[f.get(n, 0) for n in FN] for f in test_feats_list])
X_test_scaled = scaler.transform(X_test)

gb_test_preds = gb.predict(X_test_scaled)
test_preds = gb_test_preds + test_raw_boosts
test_mae = mean_absolute_error(test_targets, test_preds)

print(f'测试集 MAE: {test_mae:.4f}')
print(f'测试集 boost 范围: [{test_raw_boosts.min():.2f}, {test_raw_boosts.max():.2f}]')
print(f'测试集 GB 输出范围: [{gb_test_preds.min():.2f}, {gb_test_preds.max():.2f}]')

# ========== 9. 按区间统计测试集 ==========
print(f'\n===== 测试集按区间 MAE =====')
for lo, hi in [(13,14),(14,15),(15,16),(16,17),(17,20)]:
    mask = (test_targets >= lo) & (test_targets < hi)
    if mask.sum() == 0: continue
    m = mean_absolute_error(test_targets[mask], test_preds[mask])
    print(f'  [{lo},{hi}) n={mask.sum():2d}  MAE={m:.4f}')

# ========== 10. 全量训练 (用于最终部署) ==========
print(f'\n===== 全量训练 (用于部署) =====')
all_targets = np.array([f['_difficulty'] for f in all_charts])
all_raw_boosts = np.array([_dc(compute_raw_boost(f)) for f in all_charts])
all_gb_targets = all_targets - all_raw_boosts

X_all = np.array([[f.get(n, 0) for n in FN] for f in all_charts])
scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X_all)

gb_all = GradientBoostingRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    min_samples_leaf=3, subsample=0.8, random_state=42
)
gb_all.fit(X_all_scaled, all_gb_targets)

all_preds = gb_all.predict(X_all_scaled) + all_raw_boosts
all_mae = mean_absolute_error(all_targets, all_preds)
print(f'全量训练集 MAE: {all_mae:.4f}')

# ========== 11. 保存模型 ==========
model = {
    'gb': gb_all,
    'scaler': scaler_all,
    'feature_names': FN,
    'FLAT_FEATURES': FLAT,
    'p95_vals': P95,
    'p99_vals': P99,
    'dynamic_cap': DC,
    'version': '8.7',
    'description': '正确分割 + Boost 主导 + GB 正则化'
}
os.makedirs('models', exist_ok=True)
with open('models/6dim_model_v8_7.pkl', 'wb') as f:
    pickle.dump(model, f)
print(f'\nSaved: models/6dim_model_v8_7.pkl ({len(FLAT)} boost features)')

# ========== 12. 极端谱面诊断 ==========
print(f'\n===== 极端谱面诊断 (测试集) =====')
errors = test_preds - test_targets
extreme_idx = np.where(np.abs(errors) > 0.5)[0]
print(f'{"Name":<30s} {"True":>6s} {"Boost":>7s} {"GB":>7s} {"Pred":>7s} {"Err":>7s}')
for i in extreme_idx:
    name = test_charts[i]['_name'][:28]
    print(f'{name:<30s} {test_targets[i]:>6.1f} {test_raw_boosts[i]:>7.2f} {gb_test_preds[i]:>7.2f} {test_preds[i]:>7.2f} {errors[i]:>+7.2f}')

print(f'\n===== 完成 =====')