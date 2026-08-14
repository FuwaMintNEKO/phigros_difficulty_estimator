import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, numpy as np
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

print('='*60)
print('  GB(198) + 归一化多押外推')
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

p95_vals = {}
for j, name in enumerate(feature_names):
    p95_vals[name] = np.percentile(X[:, j], 95) if np.max(X[:, j]) > 0 else 0

bins = np.digitize(y, bins=[0,5,7,9,11,13,14,15,16,16.5,17])
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(sss.split(X, bins))
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X[train_idx])
X_te_s = scaler.transform(X[test_idx])
y_tr, y_te = y[train_idx], y[test_idx]
lv_te = lv_arr[test_idx]

print('\n--- GB ---')
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
    print(f'    {lv} ({np.sum(mask)}个): R²={r2_score(y_t,y_p):.4f}, MAE={mean_absolute_error(y_t,y_p):.4f}')

scaler_full = StandardScaler()
X_full = scaler_full.fit_transform(X)
gb_full = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
gb_full.fit(X_full, y)

# ====== 外推公式 ======
# 使用归一化多押率：mf / hs_ratio
# 设计思路：一首谱的"多押负担" = 多押事件数 / 手速超出程度
# 只有同时具备高手速+高多押的谱，hf_index才大
def compute_boost(feats, p95):
    hs = feats.get('hand_speed_index', 0)
    hs_p95 = p95.get('hand_speed_index', 1)
    mf = feats.get('multi_finger_3plus_events', 0)
    
    if hs_p95 <= 0 or hs <= hs_p95:
        return 0.0
    
    hs_ratio = hs / hs_p95
    hs_boost = 0.50 * float(np.log1p(hs_ratio - 1))
    hs_boost = min(hs_boost, 0.6)
    
    # 归一化多押率：每单位手速对应的多押事件数
    # Chart_SP #13: mf=123, hs_ratio=2.03 → hf=60.6 → mf_boost=0.485
    # Chart_SP: mf=48, hs_ratio=2.04 → hf=23.5 → mf_boost=0.188
    # Aether: mf=1, hs_ratio=1.42 → hf=0.7 → mf_boost=0.006
    # Regrets: mf=161, hs_ratio=2.05 → hf=78.5 → mf_boost=0.628
    hf_ratio = mf / max(hs_ratio, 1.0)
    mf_boost = 0.008 * hf_ratio
    mf_boost = min(mf_boost, 0.8)
    
    total = hs_boost + mf_boost
    return min(total, 2.5)

# ====== 测试谱面 ======
print('\n' + '='*60)
print('  测试谱面')
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
    p_gb = float(gb_full.predict(xs)[0])
    p_boost = compute_boost(feats, p95_vals)
    p_final = p_gb + p_boost
    
    hs = feats.get('hand_speed_index', 0)
    mf = feats.get('multi_finger_3plus_events', 0)
    hs_p95 = p95_vals.get('hand_speed_index', 1)
    hs_ratio = hs / max(hs_p95, 1)
    hf_ratio = mf / max(hs_ratio, 1) if hs_ratio > 0 else 0
    
    meta = f' ({raw["META"]["level"]})' if is_rpe else ''
    print(f'\n  {name}{meta}:')
    print(f'    GB={p_gb:.2f} + Boost={p_boost:.3f} = {p_final:.2f}')
    print(f'    hs_ratio={hs_ratio:.2f}  mf={mf}  hf_ratio(归一化多押率)={hf_ratio:.1f}')
    results[name] = {'gb': p_gb, 'boost': p_boost, 'final': p_final,
                     'hs_ratio': hs_ratio, 'mf': mf, 'hf_ratio': hf_ratio}

# ====== 保存 ======
model_data = {
    'gb': gb_full, 'scaler': scaler_full, 'feature_names': feature_names,
    'p95_vals': p95_vals,
    'metrics': {'r2': r2_gb, 'mae': mae_gb}
}
save_path = os.path.join(os.path.dirname(__file__), 'models', 'gb_final_model.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(model_data, f)

print('\n' + '='*60)
print('  结果总览')
print('='*60)
print(f'  GB内推: R²={r2_gb:.4f}, MAE={mae_gb:.4f}')
for name, r in results.items():
    print(f'  {name}: {r["final"]:.2f} (GB={r["gb"]:.2f}, boost={r["boost"]:.3f}, '
          f'hf_ratio={r["hf_ratio"]:.1f})')
