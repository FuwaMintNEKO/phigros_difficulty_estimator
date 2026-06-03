import os, sys, json, pickle, numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features

CHART_DIR = r'D:\迅雷下载\Phigros_Resource-master\Phigros_Resource-master\chart'
DIFFICULTY_TSV = r'D:\迅雷下载\Phigros_Resource-master\Phigros_Resource-master\info\difficulty.tsv'
sys.path.insert(0, os.path.dirname(__file__))
from unified_parser import load_chart

print('='*70)
print('  Phigros 难度预测系统 v5.3（指数cap）')
print('  训练集: 官谱957 (无自定义谱)')
print('  测试集: test_datas+Downloads 共20张谱面')
print('  密度使用 tap+hold (core_notes) 替代全音符')
print('='*70)

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

DOWNLOADS = r'C:\Users\NaNK\Downloads'

# 训练集仅使用官谱，无自定义谱
custom_charts = []

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
    fp = os.path.join(DOWNLOADS, path_suffix)
    try:
        cd = load_chart(fp)
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

# ====== Boost设计 ======
def _compute_dim_boost(feats, p95, p99, feat_list):
    raw = 0.0
    for fname, baseline, coeff in feat_list:
        val = feats.get(fname, 0)
        # 使用 P95*0.65 作为阈值（~P60水平），让中高特征值也能贡献boost
        pv = p95.get(fname, 0)
        thresh = max(pv * 0.55, baseline * 0.5)
        if val <= thresh:
            continue
        excess = val / thresh - 1.0
        contrib = coeff * (excess ** 0.55)
        if val > max(p99.get(fname, 0), baseline * 0.5):
            p99_excess = val / max(p99.get(fname, 0), baseline * 0.5) - 1.0
            p99_bonus = coeff * max(0, p99_excess) ** 0.55 * 0.5
            contrib += p99_bonus
        raw += contrib
    return raw

# ====== 平铺特征定义（模块级别，供保存和复用） ======
FLAT_FEATURES = [
    # === 密度 (Density — 核心: tap+hold) ===
    ('core_notes_per_second', 3.0, 0.12),
    ('notes_per_second', 3.0, 0.04),  # 全音符密度退化，辅助贡献
    ('peak_density_top5avg_1beat', 0.5, 0.15),
    ('density_above_zero_ratio', 0.6, 0.08),
    ('std_density_1beat', 0.3, 0.08),
    
    # === 1smax密度 (1s Max Density — 核心: tap+hold) ===
    ('core_peak_density_1sec_top5avg', 8, 0.20),
    ('peak_density_1sec_top5avg', 8, 0.06),  # 全音符峰值退化
    ('peak_tps_1sec_top5avg', 8, 0.15),
    ('micro_peak_top5_0.0625beat', 0.5, 0.08),
    
    # === 平均位移 (Movement) ===
    ('movement_per_second', 3.0, 0.12),
    ('burst_avg_movement', 0.5, 0.08),
    ('wide_jump_density', 0.5, 0.08),
    ('sim_pos_spread_max', 3, 0.06),
    
    # === 耐力 (Stamina) ===
    ('stamina_ratio', 0.3, 0.30),
    ('tap_per_second', 2.5, 0.20),
    ('total_notes', 400, 0.10),
    ('tap_count', 400, 0.06),
    ('duration_sec', 100, 0.06),
    ('global_jack_count', 20, 0.06),
    ('burst_intensity_mean', 0.3, 0.08),
    ('tap_burst_top5', 0.5, 0.12),
    
    # === 读谱 (Reading) ===
    ('density_transition_mean', 0.15, 0.12),
    ('density_transition_std', 0.2, 0.08),
    ('tempo_change_count', 50, 0.12),
    ('offbeat_ratio', 0.04, 0.08),
    ('rhythm_entropy', 2.5, 0.06),
    ('type_switch_per_sec', 0.4, 0.06),
    ('multi_finger_3plus_events', 10, 0.06),
]

# 动态cap参数（供 _dynamic_cap 使用）
DC = {'knee': 2.5, 'power': 0.9}


def compute_simple_boost(feats, p95, p99):
    """平铺高相关特征列表 — 不区分维度，全部放在一块算""" 
    """26特征全面平铺，按5大类别分组"""
    total_boost = _compute_dim_boost(feats, p95, p99, FLAT_FEATURES)
    total_boost = _dynamic_cap(total_boost)

    return total_boost, {'total_boost': round(total_boost, 4)}


def _dynamic_cap(raw):
    """指数衰减cap：线性到knee，超出部分 ^power 加上去，无硬上限"""
    KNEE = DC['knee']; POWER = DC['power']
    if raw <= KNEE:
        return raw
    excess = raw - KNEE
    return KNEE + excess ** POWER

# ====== 联合训练GB+boost ======
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

print('  计算boost...')
all_boosts = np.array([compute_simple_boost(feats_list[i], p95_vals, p99_vals)[0] for i in range(n_samples)])
train_boosts = all_boosts[train_idx]
test_boosts = all_boosts[test_idx]

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
print(f'  测试集: R2={r2:.4f}, MAE={mae:.4f}')

# 全量训练
X_all_s = scaler_gb.fit_transform(X_full)
y_all_residual = y_full - all_boosts
gb_full = GradientBoostingRegressor(n_estimators=600, max_depth=5, min_samples_leaf=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
gb_full.fit(X_all_s, y_all_residual)
print(f'  全量训练完成 (n={n_samples})')

# ====== 全量评估 ======
print('\n' + '='*70)
print('  训练集内评估')
print('='*70)

for i in range(n_samples):
    x = np.array([[feats_list[i].get(n,0) for n in feature_names]])
    xs = scaler_gb.transform(x)
    p_gb = float(gb_full.predict(xs)[0])
    p_b, dims = compute_simple_boost(feats_list[i], p95_vals, p99_vals)
    p_f = p_gb + p_b
    print(f'{names_list[i]:<35} 真={labels[i]:.1f}  GB={p_gb:.3f}  +Boost={p_b:.3f}  ={p_f:.3f}  [{p_f-labels[i]:+.3f}]')

# ====== 保存模型 ======
model_out = {
    'gb': gb_full, 'scaler': scaler_gb, 'feature_names': feature_names,
    'p95_vals': p95_vals, 'p99_vals': p99_vals,
    'FLAT_FEATURES': FLAT_FEATURES,
    'dynamic_cap': {'knee': 2.5, 'power': 0.9},
    'metrics': {'r2': r2, 'mae': mae, 'n_train': n_samples},
}
out_path = os.path.join(os.path.dirname(__file__), 'models', '5dim_model_v5_3.pkl')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'wb') as f:
    pickle.dump(model_out, f)
print(f'\n  模型已保存: {out_path}')

# 保存CSV
import csv
csv_path = out_path.replace('.pkl', '_predictions_v5_3.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['谱面', '难度', 'GB', 'Boost', '预测', '误差'])
    for i in range(n_samples):
        x = np.array([[feats_list[i].get(n,0) for n in feature_names]])
        xs = scaler_gb.transform(x)
        p_gb = float(gb_full.predict(xs)[0])
        p_b, dims = compute_simple_boost(feats_list[i], p95_vals, p99_vals)
        p_f = p_gb + p_b
        w.writerow([names_list[i], labels[i], round(p_gb,3), round(p_b,3), round(p_f,3), round(p_f-labels[i],3)])
print(f'  预测CSV已保存: {csv_path}')
print('\n' + '='*70)
print('  训练完成!')
print('='*70)
