import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
sys.path.insert(0, os.path.dirname(__file__))
from predict_rpe import convert_rpe_to_standard

print('='*60)
print('  GB外推修正模型')
print('='*60)

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
print(f'样本: {len(all_items)}')

feats_list, labels, levels_list = [], [], []
for i, item in enumerate(all_items):
    try:
        cd = load_chart_json(item['filepath'])
        feats = extract_features(cd)
        if feats:
            feats_list.append(feats)
            labels.append(item['difficulty'])
            levels_list.append(item['level'])
    except: pass
    if (i+1)%300==0: print(f'  {i+1}/{len(all_items)}')

feature_names = sorted(feats_list[0].keys())
X = np.array([[f.get(n,0) for n in feature_names] for f in feats_list])
y = np.array(labels)
lv_arr = np.array(levels_list)
print(f'  成功: {len(feats_list)}, 特征: {len(feature_names)}, 难度: {y.min():.1f}~{y.max():.1f}')

# 训练集P95值 - 用于检测外推
p95_vals = {}
for j, name in enumerate(feature_names):
    p95_vals[name] = np.percentile(X[:, j], 95) if np.max(X[:, j]) > 0 else 0

# 分层split + 评估
bins = np.digitize(y, bins=[0,5,7,9,11,13,14,15,16,16.5,17])
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(sss.split(X, bins))
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X[train_idx])
X_te_s = scaler.transform(X[test_idx])
y_tr, y_te = y[train_idx], y[test_idx]
lv_te = lv_arr[test_idx]

# ====== 训练GB ======
print('\n--- GB (最优基模型) ---')
gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                learning_rate=0.05, subsample=0.8, random_state=42)
gb.fit(X_tr_s, y_tr)
y_pred_gb = gb.predict(X_te_s)
r2_gb = r2_score(y_te, y_pred_gb)
mae_gb = mean_absolute_error(y_te, y_pred_gb)
print(f'  测试集: R²={r2_gb:.4f}, MAE={mae_gb:.4f}')

for lv in ['EZ','HD','IN','AT']:
    mask = lv_te == lv
    if np.sum(mask)<3: continue
    y_t, y_p = y_te[mask], y_pred_gb[mask]
    print(f'    {lv} ({np.sum(mask)}个): R²={r2_score(y_t,y_p):.4f}, MAE={mean_absolute_error(y_t,y_p):.4f}, 偏差={np.mean(y_p-y_t):+.3f}, ±0.1={np.mean(np.abs(y_p-y_t)<=0.1)*100:.0f}%')

# ====== 用全部数据训练最终模型 ======
scaler_full = StandardScaler()
X_full = scaler_full.fit_transform(X)
gb_full = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
gb_full.fit(X_full, y)
print(f'\n  全量训练完成')

# ====== 手动外推层 ======
# 对特征值超出训练P95的chart，根据超出比例给额外boost
# 设计原则：
#   - 内推不变（特征都在P95以内）：不加boost
#   - 外推适量（特征略超P95）：轻微boost
#   - 极端外推（特征远超P95）：较大boost但有限制

# 定义"手速相关"特征（用于外推判断）
SPEED_FEATURES = [
    'hand_speed_index', 'tap_per_second', 'notes_per_second',
    'core_micro_top5_0.125beat', 'core_micro_top5_0.25beat',
    'micro_max_0.0625beat', 'peak_density_16beat',
    'mean_density_16beat', 'tap_burst_top5', 'tap_burst_peak_to_mean',
    'std_density_1beat', 'std_density_2beat',
    'sustained_density_run_count', 'tempo_change_count',
]

def _safe_ratio(v, p95):
    """计算特征超出P95的比例，处理P95=0的情况"""
    if p95 <= 0:
        if v <= 0:
            return 0.0
        # P95=0意味着95%的训练集该特征=0
        # 用固定基线5，避免微小异常值过度膨胀
        baseline = 5.0
    else:
        baseline = p95
    if v <= baseline:
        return 0.0
    return v / baseline


def compute_extrapolation_boost(feats, p95_ref, speed_features, p_base=0.0):
    """多因子外推boost
    设计原则：
    - 仅对GB基预测≥16的谱面激活（避免影响中低难度谱面）
    - hand_speed提供基础boost
    - 多押特征只加成真正的多押谱
    校准: Chart_SP #13 (GB=16.48) → 17.7 (需boost≈1.22)
          Chart_SP (GB=16.39) → ~17.0 (hand_speed主导)
          Aether (GB=16.47) → ~16.6 (适度)
    """
    if p_base < 16.0:
        return 0.0

    boost = 0.0
    
    # 1. 手速指数 - 所有高难谱的基础boost
    hs = feats.get('hand_speed_index', 0)
    r = _safe_ratio(hs, p95_ref.get('hand_speed_index', 0))
    if r > 0:
        boost += min(0.50 * np.log1p(r - 1), 0.6)
    
    # 2. 同时多押 - 只有多押谱才激活
    mf = feats.get('multi_finger_3plus_events', 0)
    r = _safe_ratio(mf, p95_ref.get('multi_finger_3plus_events', 0))
    if r > 0:
        boost += min(0.15 * np.log1p(r - 1), 0.6)
    
    # 3. 错位多押（时域密集）
    tmp = feats.get('temporal_multi_press_events', 0)
    r = _safe_ratio(tmp, p95_ref.get('temporal_multi_press_events', 0))
    if r > 0:
        boost += min(0.06 * np.log1p(r - 1), 0.3)
    
    # 4. 楼梯跑阶
    sc = feats.get('staircase_count', 0)
    r = _safe_ratio(sc, p95_ref.get('staircase_count', 0))
    if r > 0:
        boost += min(0.06 * np.log1p(r - 1), 0.3)
    
    # 5. 分段最大密度 - 小权重，仅对极端密度段加成
    sd = feats.get('section_density_max', 0)
    r = _safe_ratio(sd, p95_ref.get('section_density_max', 0))
    if r > 0:
        boost += min(0.04 * np.log1p(r - 1), 0.25)
    
    # 6. 分段最大Tap密度
    st = feats.get('section_tap_density_max', 0)
    r = _safe_ratio(st, p95_ref.get('section_tap_density_max', 0))
    if r > 0:
        boost += min(0.06 * np.log1p(r - 1), 0.25)
    
    # 7. 密集多押burst
    mb = feats.get('mf_burst_count', 0)
    r = _safe_ratio(mb, p95_ref.get('mf_burst_count', 0))
    if r > 0:
        boost += min(0.06 * np.log1p(r - 1), 0.3)
    
    return min(boost, 3.0)

# ====== 测试谱面 ======
print('\n' + '='*60)
print('  测试谱面预测')
print('='*60)

test_charts = [
    ('Chart_SP', os.path.join(_ROOT, 'data', 'chart', 'Chart_SP.json'), False),
    ('Chart_SP #13', os.path.join(_ROOT, 'data', 'chart', 'Chart_SP #1347(1).json'), False),
    ('Regrets', os.path.join(_ROOT, 'data', 'chart', 'Sigma (Haocore Mix) ~ Regrets of The Yellow Tuli.json'), False),
    ('105秒伝說', os.path.join(_ROOT, 'data', 'chart', 'Sigma (Haocore Mix) ~ 105秒の伝說 ~.json'), False),
    ('Aether Crest', os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json'), True),
]

results = {}
for name, path, is_rpe in test_charts:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    cd = convert_rpe_to_standard(raw) if is_rpe else raw
    feats = extract_features(cd)
    if not feats: continue
    
    x = np.array([[feats.get(n,0) for n in feature_names]])
    xs = scaler_full.transform(x)
    p_base = float(gb_full.predict(xs)[0])
    p_boost = compute_extrapolation_boost(feats, p95_vals, SPEED_FEATURES, p_base)
    p_final = p_base + p_boost
    
    meta = f' ({raw["META"]["level"]})' if is_rpe else ''
    print(f'\n  {name}{meta}:')
    print(f'    GB基础={p_base:.2f} + 外推Boost={p_boost:.3f} = {p_final:.2f}')
    
    # 显示极端特征和新特征
    extreme_list = []
    for sf in SPEED_FEATURES:
        if sf in feats and sf in p95_vals and p95_vals[sf] > 0:
            v = feats[sf]
            p95 = p95_vals[sf]
            if v > p95 * 1.2:
                extreme_list.append(f'{sf}={v:.1f}(P95={p95:.1f})')
    if extreme_list:
        for e in extreme_list[:6]:
            print(f'    超出特征: {e}')
    
    # 显示关键新特征
    new_key_feats = ['multi_finger_3plus_events','multi_finger_4plus_events','mf_burst_count',
                     'staircase_count','staircase_total_notes','temporal_multi_press_events',
                     'temporal_multi_press_max_consecutive','section_density_max',
                     'section_tap_density_max','section_peak_to_mean_ratio']
    new_vals = [f'{k}={feats.get(k,0):.1f}' for k in new_key_feats if feats.get(k,0) > 0]
    if new_vals:
        print(f'    新特征: {", ".join(new_vals[:6])}')
    
    results[name] = {'base': p_base, 'boost': p_boost, 'final': p_final}

# =============================================
# 保存模型（含外推层参数）
# =============================================
model_data = {
    'gb': gb_full, 'scaler': scaler_full, 'feature_names': feature_names,
    'p95_ref': p95_vals, 'speed_features': SPEED_FEATURES,
    'metrics': {'r2': r2_gb, 'mae': mae_gb}
}
save_path = os.path.join(os.path.dirname(__file__), 'models', 'gb_extrapolation_model.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(model_data, f)

print('\n' + '='*60)
print('  结果总览')
print('='*60)
print(f'  GB内推: R²={r2_gb:.4f}, MAE={mae_gb:.4f}')
for name, r in results.items():
    print(f'  {name}: {r["final"]:.2f} (基础={r["base"]:.2f}, 外推={r["boost"]:.3f})')
print(f'\n  保存: {save_path}')
