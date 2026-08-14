import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, copy, numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
sys.path.insert(0, os.path.dirname(__file__))
from predict_rpe import convert_rpe_to_standard

print('='*70)
print('  Phigros 5维度难度预测系统 v3.2')
print('  纯官方训练 + 强化boost外推')
print('  【不加入任何自制谱到训练集】')
print('='*70)

# ====== 1. 加载官方数据 ONLY ======
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
print(f'\n官方谱面: {len(all_items)}')

feats_list, labels, levels_list, names_list = [], [], [], []
for i, item in enumerate(all_items):
    try:
        cd = load_chart_json(item['filepath'])
        feats = extract_features(cd)
        if feats:
            feats_list.append(feats)
            labels.append(item['difficulty'])
            levels_list.append(item['level'])
            names_list.append(item['folder'])
    except: pass
    if (i+1)%300==0: print(f'  加载 {i+1}/{len(all_items)}')

print(f'  成功提取: {len(feats_list)}')

feature_names = sorted(feats_list[0].keys())
X_full = np.array([[f.get(n,0) for n in feature_names] for f in feats_list])
y_full = np.array(labels)
print(f'特征: {len(feature_names)}, 难度: {y_full.min():.1f}~{y_full.max():.1f}')

# P95/P99
p95_vals, p99_vals = {}, {}
for j, name in enumerate(feature_names):
    col = X_full[:, j]
    p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
    p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0

# ====== 2. 训练GB (官方only) ======
print('\n--- 训练GB (仅官方957张) ---')
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
bins = np.digitize(y_full, bins=[0,5,7,9,11,13,14,15,16,16.5,17])
train_idx, test_idx = next(sss.split(X_full, bins))

scaler_gb = StandardScaler()
X_tr_s = scaler_gb.fit_transform(X_full[train_idx])
X_te_s = scaler_gb.transform(X_full[test_idx])
y_tr, y_te = y_full[train_idx], y_full[test_idx]

gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                learning_rate=0.05, subsample=0.8, random_state=42)
gb.fit(X_tr_s, y_tr)
y_pred_gb = gb.predict(X_te_s)
print(f'  测试集: R²={r2_score(y_te, y_pred_gb):.4f}, MAE={mean_absolute_error(y_te, y_pred_gb):.4f}')

X_all_s = scaler_gb.fit_transform(X_full)
gb_full = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
gb_full.fit(X_all_s, y_full)
print(f'  全量训练完成 (n={len(y_full)})')

# ====== 3. v3.2 Boost: P95线性触发 + P99倍率器 + 纯外推 ======
def _dim_boost(feats, p95, p99, feat_list):
    """从P95开始触发, 超过P99的有额外加成, power=0.6"""
    raw = 0.0
    for fname, baseline, coeff in feat_list:
        val = feats.get(fname, 0)
        thresh = max(p95.get(fname, 0), baseline)
        if val <= thresh: continue
        excess = val / thresh - 1.0
        contrib = coeff * (excess ** 0.6)
        if val > max(p99.get(fname, 0), baseline):
            p99_excess = val / max(p99.get(fname, 0), baseline) - 1.0
            contrib += coeff * max(0, p99_excess) ** 0.6 * 0.5
        raw += contrib
    return raw

def compute_5dim_boost(feats, p95, p99):
    """只靠boost外推, 不依赖训练集"""
    total_n = max(feats.get('total_notes', 1), 1)

    d1f = [
        ('tap_micro_max_0.0625beat',  2.0,  0.55),
        ('tap_micro_top5_0.0625beat',  1.2,  0.40),
        ('tap_burst_top5',             6.0,  0.35),
        ('jack_count',                20.0,  0.30),
        ('tap_per_second',             4.2,  0.30),
        ('very_short_interval_ratio',  0.18, 0.25),
        ('tap_burst_05_top5',          4.0,  0.35),
    ]
    d1 = _dim_boost(feats, p95, p99, d1f)

    mf3 = feats.get('multi_finger_3plus_events', 0)
    spread_max = feats.get('sim_pos_spread_max', 0)
    fmi = mf3 * spread_max / max(total_n, 1) * 10
    d2 = 0.0
    th = max(p99.get('multi_finger_3plus_events', 30), 1) * max(p99.get('sim_pos_spread_max', 0.8), 0.1) / max(p99.get('total_notes', 500), 1) * 10
    if fmi > max(th * 0.5, 0.3):
        d2 = 0.50 * ((fmi / max(th * 0.5, 0.3) - 1) ** 0.6)

    d3f = [
        ('wide_jump_count',            60.0,  0.40),
        ('burst_avg_movement',          1.5,  0.30),
        ('hold_lock_displacement_per_sec', 0.8, 0.40),
        ('movement_per_second',         7.0,  0.12),
        ('hold_lock_tap_events_per_hold', 1.0, 0.25),
    ]
    d3 = _dim_boost(feats, p95, p99, d3f)

    d4f = [
        ('total_notes',               800.0,  0.45),
        ('tap_notes_per_second',       5.0,   0.35),
        ('notes_per_second',           7.5,   0.15),
        ('high_density_duration_ratio_16beat', 0.15, 0.20),
        ('sustained_density_run_count', 1.0,  0.18),
    ]
    d4 = _dim_boost(feats, p95, p99, d4f)

    d5f = [
        ('density_transition_max',      2.5,  0.75),
        ('tempo_change_count',         30.0,  0.55),
        ('speed_change_total_impact', 20000,  0.28),
        ('offbeat_ratio',              0.08,  0.30),
        ('rhythm_entropy',             3.0,   0.15),
        ('bpm_change_count',            0.5,  0.30),
        ('density_transition_mean',     0.30, 0.38),
        ('type_switch_ratio',           0.06, 0.22),
        ('type_switch_per_sec',         0.8,  0.18),
    ]
    d5 = _dim_boost(feats, p95, p99, d5f)

    total_boost = d1 * 0.22 + d2 * 0.10 + d3 * 0.18 + d4 * 0.18 + d5 * 0.30
    total_boost = min(total_boost, 3.0)

    return total_boost, {'dim1_交互纵连': round(d1, 4), 'dim2_多押': round(d2, 4),
                          'dim3_位移': round(d3, 4), 'dim4_耐力': round(d4, 4),
                          'dim5_读谱': round(d5, 4)}

# ====== 4. 测试自制谱 (全部未参与训练!) ======
print('\n' + '='*70)
print('  自制谱测试 (全部未参与训练)')
print('='*70)

custom_list = [
    ('LiFE Garden', '6923526264684294.json', 'AT Lv.18'),
    ('哀狱炼歌', '1321664301929799.json', 'AT Lv.19'),
    ('DA\'AT', '2155734445357448.json', 'AT Lv.17'),
    ('LAMIA', '29834645.json', 'AT Lv.18'),
    ('He Asked If I Exercise', '81816997.json', 'AT Lv.18'),
    ('Waking Shadows', '93562988.json', 'AT Lv.18'),
    ('Aether Crest', '4641132726938698.json', 'SP'),
    ('Chart_SP', 'Chart_SP.json', 'SP'),
    ('Chart_SP #13', 'Chart_SP #1347(1).json', 'SP'),
    ('Regrets', 'Sigma (Haocore Mix) ~ Regrets of The Yellow Tuli.json', 'SP'),
    ('105秒伝說', 'Sigma (Haocore Mix) ~ 105秒の伝說 ~.json', 'SP'),
]

all_custom_results = []
for name, path_suffix, level in custom_list:
    fp = os.path.join(CHART_DIR, path_suffix)
    with open(fp, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    is_rpe = any(s in path_suffix for s in ['46411327','21557344','29834645','81816997','93562988','13216643','69235262'])
    cd = convert_rpe_to_standard(raw) if is_rpe else raw
    feats = extract_features(cd)
    if not feats: continue
    x = np.array([[feats.get(n,0) for n in feature_names]])
    xs = scaler_gb.transform(x)
    p_gb = float(gb_full.predict(xs)[0])
    p_boost, dims = compute_5dim_boost(feats, p95_vals, p99_vals)
    p_final = p_gb + p_boost
    all_custom_results.append((p_final, name, level, p_gb, p_boost, dims, feats))

for pred, name, level, gb_p, boost, dims, feats in sorted(all_custom_results, key=lambda x: -x[0]):
    print(f'\n{name:30s} ({level})')
    print(f'  GB={gb_p:.3f} + Boost={boost:.4f} = {pred:.3f}')
    print(f'  D1={dims["dim1_交互纵连"]:.3f} D2={dims["dim2_多押"]:.3f} D3={dims["dim3_位移"]:.3f} D4={dims["dim4_耐力"]:.3f} D5={dims["dim5_读谱"]:.3f}')
    for k in ['total_notes','tap_count','drag_count','notes_per_second','tap_notes_per_second',
              'jack_count','wide_jump_count','multi_finger_3plus_events','tempo_change_count',
              'density_transition_max','offbeat_ratio','tap_micro_max_0.0625beat','tap_burst_top5']:
        v = feats.get(k, 0)
        if v > 0:
            f = ' ↑↑' if v > p99_vals.get(k, 0) else (' ↑' if v > p95_vals.get(k, 0) else '')
            print(f'    {k:35s} = {str(v):>8s}  P99={p99_vals.get(k,0):>7.2f}{f}')

# ====== 5. Rrhar BPM验证 ======
print('\n' + '='*70)
print('  Rrhar\'il BPM缩放验证')
print('='*70)
rrhar_path = os.path.join(CHART_DIR, 'Rrharil.TeamGrimoire.0', 'AT.json')
with open(rrhar_path, 'r', encoding='utf-8') as f:
    rrhar_orig = json.load(f)

for scale in [1.0, 1.1, 1.2]:
    rrhar_mod = copy.deepcopy(rrhar_orig)
    for line in rrhar_mod.get('judgeLineList', []):
        if 'bpm' in line: line['bpm'] = round(line['bpm'] * scale, 4)
    if 'META' in rrhar_mod and 'BPM' in rrhar_mod['META']:
        rrhar_mod['META']['BPM'] = round(rrhar_mod['META']['BPM'] * scale, 4)
    feats = extract_features(rrhar_mod)
    x = np.array([[feats.get(n,0) for n in feature_names]])
    xs = scaler_gb.transform(x)
    p_gb = float(gb_full.predict(xs)[0])
    p_boost, dims = compute_5dim_boost(feats, p95_vals, p99_vals)
    p_final = p_gb + p_boost
    print(f'  ×{scale:.1f}: GB={p_gb:.3f} + Boost={p_boost:.4f} = {p_final:.3f}')
    print(f'    D1={dims["dim1_交互纵连"]:.3f} D3={dims["dim3_位移"]:.3f} D4={dims["dim4_耐力"]:.3f} D5={dims["dim5_读谱"]:.3f}')

# 官方高难评估
print('\n' + '='*70)
print('  官方高难谱评估 (验证模型没把官方谱搞坏)')
print('='*70)
for name, folder, json_file in [
    ('Rrhar\'il AT', 'Rrharil.TeamGrimoire.0', 'AT.json'),
    ('QZKago AT', 'QZKagoRequiem.tpazolite.0', 'AT.json'),
    ('Distorted Fate AT', 'DistortedFate.Sakuzyo.0', 'AT.json'),
    ('Destruction 3,2,1 AT', 'Destruction321.黒皇帝.0', 'AT.json'),
    ('玩具狂奏曲 AT', '玩具狂奏曲終焉.きくお.0', 'AT.json'),
    ('Pragmatism AT', 'PRAGMATISMRESURRECTION.Laur.0', 'AT.json'),
    ('Slips AT', 'slips.rintarosoma.0', 'AT.json'),
]:
    fp = os.path.join(CHART_DIR, folder, json_file)
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        feats = extract_features(raw)
        x = np.array([[feats.get(n,0) for n in feature_names]])
        xs = scaler_gb.transform(x)
        p_gb = float(gb_full.predict(xs)[0])
        p_boost, dims = compute_5dim_boost(feats, p95_vals, p99_vals)
        p_final = p_gb + p_boost
        print(f'  {name:35s}  GB={p_gb:.3f} +Boost={p_boost:.3f} ={p_final:.3f}')
    except Exception as e:
        print(f'  {name}: {e}')

# 保存
model_data = {
    'gb': gb_full, 'scaler': scaler_gb, 'feature_names': feature_names,
    'p95_vals': p95_vals, 'p99_vals': p99_vals,
}
save_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, '5dim_model_v3.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f'\n  保存: {save_path}')
print('='*70)
