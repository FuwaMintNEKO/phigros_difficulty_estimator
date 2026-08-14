"""
v8.6 — 统计学证明驱动改进
==========================
基于 v8_6_stat_proof.py 的数学证明:
  H1: rcnps 与 density_dimension r=0.99 → 移除 rcnps, co 合并到 density_dimension
  H2: speed_volatility r=-0.003 → 移除
  H7: drag_flick_ratio r=-0.03, pattern_switch_rate r=-0.05 → 移除
  H4: above_avg_duration_sec r=0.73 → 加入 (co=0.05)
  H5: trill_density r=0.38, 偏r=0.17 → 加入 (co=0.03)
  额外: multi_finger_3plus_events r=0.38 → 加入 (co=0.02)
  H6: density_dimension 算术平均 r=0.8219 > 几何 r=0.8174 → 已切换公式
  H3: 配置维度再平衡 → 提升 chord_alternation, stair_rate, weighted_mf 的 co
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import pickle, os, sys, numpy as np, math
sys.path.insert(0,'.')
from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from unified_parser import load_chart_from_bytes
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print('='*60)
print('  v8.6 — 统计证明驱动改进 (P95/P99全量重算)')
print('='*60)

# 加载 v8.4 的 DC/FN 作为起点 (DC参数和GB特征名不变)
with open('models/6dim_model_v8_4.pkl', 'rb') as f:
    v84 = pickle.load(f)
DC = v84['dynamic_cap']; FNo = v84['feature_names']

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)

all_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    for lv in ['IN','AT']:
        if lv in info['levels'] and lv in song_difficulties[sid]:
            all_items.append({'folder':fn,'filepath':info['levels'][lv],'difficulty':song_difficulties[sid][lv],'level':lv})

feats_list = []
for item in all_items:
    try:
        cd = load_chart_json(item['filepath'])
        feats = extract_features(cd)
        if feats: feats_list.append(feats)
    except: pass

n_all = len(feats_list)
print(f'{n_all} charts loaded')

# ============================================================
# v8.6 FLAT_FEATURES — 基于统计证明
# ============================================================
# 移除: rcnps(r=0.99冗余), speed_volatility(r=-0.003), drag_flick_ratio(r=-0.03), pattern_switch_rate(r=-0.05)
# 新增: above_avg_duration_sec(r=0.73), trill_density(r=0.38,偏r=0.17), multi_finger_3plus_events(r=0.38)
# 重平衡: density_dimension co 0.08→0.12(合并rcnps), 配置关键特征提co
FLAT = [
    # ====== 密度 (2个, 移除了rcnps) ======
    ('density_dimension',              1.0,    0.12),   # +0.04 合并 rcnps 的 co
    # ====== 配置 (11个, 移除drag_flick_ratio和pattern_switch_rate, 新增trill_density和multi_finger_3plus) ======
    ('stair_rate_per_sec',             2.0,    0.06),   # +0.01 r=0.65
    ('stair_complexity',               0.2,    0.02),
    ('chord_size_entropy',             0.5,    0.02),
    ('chord_alternation_rate',         0.5,    0.10),   # +0.02 r=0.66
    ('weighted_mf_score_per_sec',      10.0,   0.06),   # +0.01 r=0.52
    ('position_entropy',               2.0,    0.02),
    ('avg_chord_size_poly',            2.0,    0.03),
    ('position_range_used',            0.5,    0.02),
    ('trill_density',                  2.0,    0.03),   # NEW r=0.38,偏r=0.17
    ('multi_finger_3plus_events',      0.5,    0.02),   # NEW r=0.38
    # ====== 耐力 (4个, 新增above_avg_duration_sec) ======
    ('above_avg_density_mean',         4.0,    0.25),   # 最强信号 r=0.83
    ('above_avg_duration_sec',         30.0,   0.05),   # NEW r=0.73,偏r=0.10
    ('total_notes',                    400.0,  0.15),   # 锁定: 非对称拉升
    ('tap_burst_top5',                 0.5,    0.04),
    # ====== 读谱 (11个, 移除speed_volatility) ======
    ('tempo_change_count',             50.0,   0.02),
    ('type_switch_per_sec',            0.4,    0.05),
    ('density_transition_std',         0.2,    0.04),
    ('density_transition_mean',        0.15,   0.02),
    ('note_clutter_ratio',             0.05,   0.04),
    ('rhythm_entropy',                 2.5,    0.03),
    ('hold_interference_index',        0.3,    0.04),
    ('jline_movement_density',         50.0,   0.05),
    ('jline_rotate_density',           20.0,   0.03),
    ('jline_disappear_density',        20.0,   0.03),
    ('above_below_cross',              0.3,    0.03),
    # ====== 高速音符 (6个, 锁定) ======
    ('fast_note_density_16th',         4.0,    0.08),
    ('fast_note_density_32nd',         2.0,    0.15),
    ('fast_note_density_24th',         1.0,    0.10),
    ('fast_note_density_48th',         0.5,    0.12),
    ('fast_note_density_64th',         0.3,    0.10),
    ('rhythm_type_count',              3.0,    0.10),
]

# 维度 co 总和统计
config_co = sum(c for n,_,c in FLAT if n in ['stair_rate_per_sec','stair_complexity','chord_size_entropy',
    'chord_alternation_rate','weighted_mf_score_per_sec','position_entropy','avg_chord_size_poly',
    'position_range_used','trill_density','multi_finger_3plus_events'])
stamina_co = sum(c for n,_,c in FLAT if n in ['above_avg_density_mean','above_avg_duration_sec','total_notes','tap_burst_top5'])
density_co = sum(c for n,_,c in FLAT if n in ['density_dimension'])
reading_co = sum(c for n,_,c in FLAT if n in ['tempo_change_count','type_switch_per_sec','density_transition_std',
    'density_transition_mean','note_clutter_ratio','rhythm_entropy','hold_interference_index',
    'jline_movement_density','jline_rotate_density','jline_disappear_density','above_below_cross'])
fast_co = sum(c for n,_,c in FLAT if 'fast_note' in n or 'rhythm_type_count' in n)

print(f'\n维度 co 总和:')
print(f'  密度: {density_co:.2f} (1特征)')
print(f'  配置: {config_co:.2f} (10特征, 平均 co={config_co/10:.3f})')
print(f'  耐力: {stamina_co:.2f} (4特征, 平均 co={stamina_co/4:.3f})')
print(f'  读谱: {reading_co:.2f} (11特征, 平均 co={reading_co/11:.3f})')
print(f'  高速: {fast_co:.2f} (6特征)')
print(f'  总计: {sum(c for _,_,c in FLAT):.2f} ({len(FLAT)}特征)')

# Compute P95/P99 for ALL features from current data (NOT from v8.4)
P95 = {}; P99 = {}
for fname, bl, _ in FLAT:
    vals = [f.get(fname, 0) for f in feats_list]
    if vals:
        P95[fname] = float(np.percentile(vals, 95))
        P99[fname] = float(np.percentile(vals, 99))
        if P95[fname] < 0.01: P95[fname] = bl * 0.5  # 防止 P95 为 0
        if P99[fname] < 0.01: P99[fname] = bl * 0.5
        print(f'{fname:<35s} P95={P95[fname]:6.2f} P99={P99[fname]:6.2f}')

# Train GB
FN = list(FNo)
X_gb = np.array([[f.get(n, 0) for n in FN] for f in feats_list])
labels = np.array([item['difficulty'] for item in all_items])
y = labels.copy()

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

co_arr = np.array([c for _,_,c in FLAT])

def compute_raw_boost(feats):
    raw = 0.0
    for j, (fname, bl, _) in enumerate(FLAT):
        raw += co_arr[j] * compute_excess(feats, fname, bl)
    return raw

def _dc(r):
    if r <= DC['knee']: return r
    return DC['knee'] + (r - DC['knee']) ** DC['power']

all_boosts = np.array([_dc(compute_raw_boost(f)) for f in feats_list])

sc_f = StandardScaler()
gb_f = GradientBoostingRegressor(n_estimators=700, max_depth=5, min_samples_leaf=3,
                                  learning_rate=0.05, subsample=0.8, random_state=42)
gb_f.fit(sc_f.fit_transform(X_gb), y - all_boosts)

# Save
out = {'gb': gb_f, 'scaler': sc_f, 'feature_names': list(FN),
       'p95_vals': P95, 'p99_vals': P99, 'FLAT_FEATURES': FLAT, 'dynamic_cap': DC,
       'sigmoid_params': {'target': 0.32, 'power': 0.65, 'thresh': 0.22}}
os.makedirs('models', exist_ok=True)
with open('models/6dim_model_v8_6.pkl', 'wb') as f: pickle.dump(out, f)
print(f'\nSaved: models/6dim_model_v8_6.pkl ({len(FLAT)} boost features)')

# Quick evaluation
def adjust_boost_smooth(boost, gb, target=0.32, power=0.75, thresh=0.22, steepness=25):
    if boost < 2.0: return boost
    ratio = boost / gb if gb > 0 else 0
    expected = target * gb
    if expected <= 0 or boost <= 0: return boost
    adj = expected * ((boost / expected) ** power)
    w = 1 / (1 + math.exp(-steepness * (ratio - thresh)))
    return (1 - w) * boost + w * adj
preds = []
for i, feats in enumerate(feats_list):
    gb_raw = gb_f.predict(sc_f.transform(X_gb[i:i+1]))[0]
    boost = _dc(compute_raw_boost(feats))
    adj_boost = adjust_boost_smooth(boost, gb_raw)
    pred = gb_raw + adj_boost
    preds.append(pred)

errors = np.array(preds) - labels
mae = np.mean(np.abs(errors))
print(f'\n训练集 MAE: {mae:.4f}')

# 按维度打印 top 特征
print(f'\nFLAT 特征 (按 co 降序):')
for n, b, c in sorted(FLAT, key=lambda x: -x[2]):
    print(f'  {n:<35s} bl={b:6.1f}  co={c:.4f}')