"""直接手动设置 FLAT_FEATURES co 值，跳过 Ridge 优化"""
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

print('='*50); print('  v8.5 — manual co + total_notes asymmetric boost'); print('='*50)

with open('models/6dim_model_v8_4.pkl', 'rb') as f:
    v84 = pickle.load(f)
P95o = v84['p95_vals']; P99o = v84['p99_vals']
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
print(f'{n_all} charts')

# ==== MANUAL FLAT_FEATURES ====
# 原则: density_dimension低co防止暴走, above_avg_mean高co主导区分, total_notes非对称拉升
# 所有r<0.15的特征移出或降至0.01
FLAT = [
    # 密度
    ('density_dimension',              1.0,    0.08),
    ('real_core_notes_per_second',     2.0,    0.03),
    # 配置 (保留 Ridge 学到的值)
    ('stair_rate_per_sec',             2.0,    0.05),
    ('stair_complexity',               0.2,    0.02),
    ('chord_size_entropy',             0.5,    0.02),
    ('chord_alternation_rate',         0.5,    0.08),
    ('weighted_mf_score_per_sec',      10.0,   0.05),
    ('position_entropy',               2.0,    0.02),
    ('avg_chord_size_poly',            2.0,    0.03),
    ('drag_flick_ratio',               0.2,    0.02),
    ('pattern_switch_rate',            1.0,    0.05),
    ('position_range_used',            0.5,    0.02),
    # 耐力: 高潮段焦点 + 非对称 total_notes
    ('above_avg_density_mean',         4.0,    0.25),   # 最强信号 r=0.84
    ('total_notes',                    400.0,  0.15),   # 锁定: 只拉升高物量谱
    ('rest_ratio',                     0.3,    0.00),   # 移除(r=-0.08, 负co不兼容excess)
    ('tap_burst_top5',                 0.5,    0.04),
    # 读谱
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
    ('speed_volatility',               0.1,    0.04),
    ('above_below_cross',              0.3,    0.03),
    # 高速音符 (锁定)
    ('fast_note_density_16th',         4.0,    0.08),
    ('fast_note_density_32nd',         2.0,    0.15),
    ('fast_note_density_24th',         1.0,    0.10),
    ('fast_note_density_48th',         0.5,    0.12),
    ('fast_note_density_64th',         0.3,    0.10),
    ('rhythm_type_count',              3.0,    0.10),
]

# Compute P95/P99
P95 = dict(P95o); P99 = dict(P99o)
for feat in ['above_avg_density_mean','total_notes','density_dimension']:
    vals = [f.get(feat,0) for f in feats_list]
    if vals:
        P95[feat] = float(np.percentile(vals, 95))
        P99[feat] = float(np.percentile(vals, 99))
        print(f'{feat}: P95={P95[feat]:.1f} P99={P99[feat]:.1f}')

# Train GB with these boosts
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
with open('models/6dim_model_v8_5.pkl', 'wb') as f: pickle.dump(out, f)
print(f'\nSaved: models/6dim_model_v8_5.pkl ({len(FLAT)} boost features)')
for n,b,c in FLAT:
    if any(x in n for x in ['above_avg','total_notes','density_dim','tap_burst']):
        print(f'  {n}: co={c:.4f}')
