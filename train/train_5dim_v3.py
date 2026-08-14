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
print('  Phigros 5维度难度预测系统 v3.1')
print('  GB+boost联合训练 | 强化外推 | P99放大器')
print('='*70)

# ====== 1. 加载全部数据 ======
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

# 自定义谱
custom_charts = [
    ('DA\'AT', '2155734445357448.json', 18.2, 'AT'),
    ('LAMIA', '29834645.json', 18.3, 'AT'),
    ('HeAskedIfIExercise', '81816997.json', 18.5, 'AT'),
    ('WakingShadows', '93562988.json', 17.8, 'AT'),
    ('哀狱炼歌', '1321664301929799.json', 18.6, 'AT'),
    ('Aether Crest', '4641132726938698.json', 15.7, 'SP'),
    ('Chart_SP', 'Chart_SP.json', 16.45, 'SP'),
    ('Chart_SP #13', 'Chart_SP #1347(1).json', 16.9, 'SP'),
    ('Regrets', 'Sigma (Haocore Mix) ~ Regrets of The Yellow Tuli.json', 17.2, 'SP'),
    ('105秒伝說', 'Sigma (Haocore Mix) ~ 105秒の伝說 ~.json', 16.1, 'SP'),
]

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

print(f'  官方提取: {len(feats_list)}')

for name, path_suffix, diff, lv in custom_charts:
    fp = os.path.join(CHART_DIR, path_suffix)
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        is_rpe = any(s in path_suffix for s in ['46411327','21557344','29834645','81816997','93562988','13216643'])
        cd = convert_rpe_to_standard(raw) if is_rpe else raw
        feats = extract_features(cd)
        if feats:
            feats_list.append(feats)
            labels.append(diff)
            levels_list.append(lv)
            names_list.append(f'{name}(自制)')
            print(f'  加入: {name} = {diff}')
    except Exception as e:
        print(f'  {name}失败: {e}')

feature_names = sorted(feats_list[0].keys())
X_full = np.array([[f.get(n,0) for n in feature_names] for f in feats_list])
y_full = np.array(labels)
n_samples = len(feats_list)
print(f'\n总谱面: {n_samples}, 特征: {len(feature_names)}, 难度: {y_full.min():.1f}~{y_full.max():.1f}')

# P95/P99只用官方数据
official_n = len(all_items)
official_feats = feats_list[:official_n]
p95_vals, p99_vals = {}, {}
for j, name in enumerate(feature_names):
    col = np.array([f.get(name,0) for f in official_feats])
    p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
    p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0

corrs_dict = {}
for j, name in enumerate(feature_names):
    col = X_full[:, j]
    if np.std(col) > 0 and np.std(y_full) > 0:
        corrs_dict[name] = float(np.corrcoef(col, y_full)[0, 1])
    else:
        corrs_dict[name] = 0

# ====== 3. 新Boost设计 v3.1 ======
# 核心变化:
#   - 每特征: contrib = coeff * (excess)^0.6  (用0.6次方压缩, 比sqrt弱)
#   - P99放大器: 超过P99的特征贡献翻倍
#   - 维度内: sum(contribs) 无再压缩
#   - 维度加权求和: 不用tanh, 直接线性sum
#   - 最终cap=3.0

def _compute_dim_boost(feats, p95, p99, feat_list):
    """维度boost: power=0.6压缩 + P99倍率器"""
    raw = 0.0
    for fname, baseline, coeff in feat_list:
        val = feats.get(fname, 0)
        thresh = max(p95.get(fname, 0), baseline)
        if val <= thresh:
            continue
        excess = val / thresh - 1.0
        contrib = coeff * (excess ** 0.6)
        if val > max(p99.get(fname, 0), baseline):
            p99_excess = val / max(p99.get(fname, 0), baseline) - 1.0
            p99_bonus = coeff * max(0, p99_excess) ** 0.6 * 0.5
            contrib += p99_bonus
        raw += contrib
    return raw

def compute_5dim_boost(feats, p95, p99):
    """5维度boost v3.1"""
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
    d1 = _compute_dim_boost(feats, p95, p99, d1f)

    mf3 = feats.get('multi_finger_3plus_events', 0)
    spread_max = feats.get('sim_pos_spread_max', 0)
    fmi = mf3 * spread_max / max(total_n, 1) * 10
    d2 = 0.0
    th = max(p99.get('multi_finger_3plus_events', 30), 1) * max(p99.get('sim_pos_spread_max', 0.8), 0.1) / max(p99.get('total_notes', 500), 1) * 10
    if fmi > max(th * 0.5, 0.3):
        exc = fmi / max(th * 0.5, 0.3) - 1
        d2 = 0.50 * (exc ** 0.6)

    d3f = [
        ('wide_jump_count',            60.0,  0.40),
        ('burst_avg_movement',          1.5,  0.30),
        ('hold_lock_displacement_per_sec', 0.8, 0.40),
        ('movement_per_second',         7.0,  0.12),
        ('hold_lock_tap_events_per_hold', 1.0, 0.25),
    ]
    d3 = _compute_dim_boost(feats, p95, p99, d3f)

    d4f = [
        ('total_notes',               800.0,  0.45),
        ('tap_notes_per_second',       5.0,   0.35),
        ('notes_per_second',           7.5,   0.15),
        ('high_density_duration_ratio_16beat', 0.15, 0.20),
        ('sustained_density_run_count', 1.0,  0.18),
    ]
    d4 = _compute_dim_boost(feats, p95, p99, d4f)

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
    d5 = _compute_dim_boost(feats, p95, p99, d5f)

    total_boost = d1 * 0.22 + d2 * 0.10 + d3 * 0.18 + d4 * 0.18 + d5 * 0.30
    total_boost = min(total_boost, 3.0)

    return total_boost, {'dim1_交互纵连': round(d1, 4), 'dim2_多押': round(d2, 4),
                          'dim3_位移': round(d3, 4), 'dim4_耐力': round(d4, 4),
                          'dim5_读谱': round(d5, 4)}

# ====== 4. 联合训练GB+boost ======
print('\n--- 联合训练 GB+boost ---')
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
bins = np.digitize(y_full, bins=[0,5,7,9,11,13,14,15,16,16.5,17,18])
train_idx, test_idx = next(sss.split(X_full, bins))

scaler_gb = StandardScaler()
X_tr_s = scaler_gb.fit_transform(X_full[train_idx])
X_te_s = scaler_gb.transform(X_full[test_idx])

y_tr_labels = y_full[train_idx].copy()
y_te_labels = y_full[test_idx].copy()
y_te_orig_labels = y_te_labels.copy()

# Step 1: 计算所有样本的boost
print('  计算boost...')
all_boosts = np.array([compute_5dim_boost(feats_list[i], p95_vals, p99_vals)[0] for i in range(n_samples)])
train_boosts = all_boosts[train_idx]
test_boosts = all_boosts[test_idx]

# Step 2: GB学习 true - boost (残差)
y_tr_residual = y_tr_labels - train_boosts
y_te_residual = y_te_labels - test_boosts

print(f'  Boost范围: [{all_boosts.min():.3f}, {all_boosts.max():.3f}]')
print(f'  训练集残差范围: [{y_tr_residual.min():.2f}, {y_tr_residual.max():.2f}]')

gb = GradientBoostingRegressor(n_estimators=600, max_depth=5, min_samples_leaf=3,
                                learning_rate=0.05, subsample=0.8, random_state=42)
gb.fit(X_tr_s, y_tr_residual)

y_pred_gb = gb.predict(X_te_s)
y_pred_final = y_pred_gb + test_boosts

r2 = r2_score(y_te_orig_labels, y_pred_final)
mae = mean_absolute_error(y_te_orig_labels, y_pred_final)
print(f'  测试集: R²={r2:.4f}, MAE={mae:.4f}')

# 全量训练
X_all_s = scaler_gb.fit_transform(X_full)
y_all_residual = y_full - all_boosts
gb_full = GradientBoostingRegressor(n_estimators=600, max_depth=5, min_samples_leaf=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
gb_full.fit(X_all_s, y_all_residual)
print(f'  全量训练完成 (n={n_samples})')

# ====== 5. 全量评估 ======
print('\n' + '='*70)
print('  统一难度评估')
print('='*70)

boosted_results = []
for i in range(n_samples):
    x = np.array([[feats_list[i].get(n,0) for n in feature_names]])
    xs = scaler_gb.transform(x)
    p_gb = float(gb_full.predict(xs)[0])
    p_boost, dims = compute_5dim_boost(feats_list[i], p95_vals, p99_vals)
    p_final = p_gb + p_boost
    err = p_final - labels[i]
    boosted_results.append({
        'name': names_list[i], 'level': levels_list[i],
        'true': labels[i], 'gb': p_gb, 'boost': p_boost,
        'pred': p_final, 'err': err, 'dims': dims,
    })

for lo, hi in [(15.5, 16.0), (16.0, 16.5), (16.5, 17.0), (17.0, 18.0), (18.0, 19.0)]:
    items = [r for r in boosted_results if lo <= r['true'] < hi]
    if not items: continue
    abs_errs = [abs(r['err']) for r in items]
    n = len(items)
    print(f'\n  难度 {lo:.1f}~{hi:.1f} (n={n}):')
    print(f'    MAE: {np.mean(abs_errs):.3f}, 偏差: {np.mean([r["err"] for r in items]):+.3f}')
    print(f'    ±0.2以内: {sum(1 for e in abs_errs if e<=0.2)/n*100:.0f}%')
    print(f'    ±0.5以内: {sum(1 for e in abs_errs if e<=0.5)/n*100:.0f}%')

all_high = [r for r in boosted_results if r['true'] >= 15.5]
print(f'\n  全部高难(n={len(all_high)}): MAE={np.mean([abs(r["err"]) for r in all_high]):.3f}')

# 自定义谱
print('\n' + '='*70)
print('  自定义谱面 + 参考谱面')
print('='*70)
custom_results = [r for r in boosted_results if '自制' in r['name'] or r['level'] == 'SP']
for r in sorted(custom_results, key=lambda x: -x['true']):
    d = r['dims']
    print(f'{r["name"][:30]:30s} 真={r["true"]:.1f}  GB={r["gb"]:.3f}  +Boost={r["boost"]:.3f}  ={r["pred"]:.3f}  [{r["err"]:+.3f}]')
    print(f'  D1={d["dim1_交互纵连"]:.3f} D2={d["dim2_多押"]:.3f} D3={d["dim3_位移"]:.3f} D4={d["dim4_耐力"]:.3f} D5={d["dim5_读谱"]:.3f}')

# ====== 6. Rrhar BPM验证 ======
print('\n' + '='*70)
print('  Rrhar\'il BPM缩放验证')
print('  目标: ×1.1→17.75, ×1.2→17.9')
print('='*70)
rrhar_path = os.path.join(CHART_DIR, 'Rrharil.TeamGrimoire.0', 'AT.json')
with open(rrhar_path, 'r', encoding='utf-8') as f:
    rrhar_orig = json.load(f)

for scale in [1.0, 1.1, 1.2]:
    rrhar_mod = copy.deepcopy(rrhar_orig)
    for line in rrhar_mod.get('judgeLineList', []):
        if 'bpm' in line:
            line['bpm'] = round(line['bpm'] * scale, 4)
    if 'META' in rrhar_mod and 'BPM' in rrhar_mod['META']:
        rrhar_mod['META']['BPM'] = round(rrhar_mod['META']['BPM'] * scale, 4)
    feats = extract_features(rrhar_mod)
    x = np.array([[feats.get(n,0) for n in feature_names]])
    xs = scaler_gb.transform(x)
    p_gb = float(gb_full.predict(xs)[0])
    p_boost, dims = compute_5dim_boost(feats, p95_vals, p99_vals)
    p_final = p_gb + p_boost
    note = ''
    if scale == 1.0: note = '(原始: 17.64)'
    elif scale == 1.1: note = f'(目标17.75, 差{p_final-17.75:+.3f})'
    elif scale == 1.2: note = f'(目标17.9, 差{p_final-17.9:+.3f})'
    print(f'  ×{scale:.1f}: GB={p_gb:.3f} + Boost={p_boost:.4f} = {p_final:.3f}  {note}')
    print(f'    D1={dims["dim1_交互纵连"]:.3f} D3={dims["dim3_位移"]:.3f} D4={dims["dim4_耐力"]:.3f} D5={dims["dim5_读谱"]:.3f}')

# ====== 7. 官方最高难参考 ======
print('\n' + '='*70)
print('  官方高难谱预测')
print('='*70)
# QZKago
qzk_path = os.path.join(CHART_DIR, 'QZKagoRequiem.tpazolite.0', 'AT.json')
for name, path, expected in [
    ('QZKago AT', qzk_path, 17.4),
]:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    feats = extract_features(raw)
    x = np.array([[feats.get(n,0) for n in feature_names]])
    xs = scaler_gb.transform(x)
    p_gb = float(gb_full.predict(xs)[0])
    p_boost, dims = compute_5dim_boost(feats, p95_vals, p99_vals)
    p_final = p_gb + p_boost
    print(f'  {name:30s} 真={expected:.1f}  GB={p_gb:.3f} +Boost={p_boost:.3f} ={p_final:.3f} [{p_final-expected:+.3f}]')

# ====== 8. 保存 ======
model_data = {
    'gb': gb_full, 'scaler': scaler_gb, 'feature_names': feature_names,
    'p95_vals': p95_vals, 'p99_vals': p99_vals, 'corrs_dict': corrs_dict,
}
save_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, '5dim_model_v3.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(model_data, f)

with open(os.path.join(save_dir, '5dim_predictions_v3.csv'), 'w', encoding='utf-8') as f:
    f.write('name,level,true,gb,boost,pred,err,')
    f.write('dim1_交互纵连,dim2_多押,dim3_位移,dim4_耐力,dim5_读谱\n')
    for r in boosted_results:
        d = r['dims']
        f.write(f'{r["name"]},{r["level"]},{r["true"]:.1f},{r["gb"]:.3f},{r["boost"]:.3f},'
                f'{r["pred"]:.3f},{r["err"]:.3f},{d["dim1_交互纵连"]},{d["dim2_多押"]},'
                f'{d["dim3_位移"]},{d["dim4_耐力"]},{d["dim5_读谱"]}\n')

print(f'\n  保存: {save_path}')
print('='*70)
