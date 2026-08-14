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
from feature_extractor import extract_features, collect_all_notes, NOTE_HOLD

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
sys.path.insert(0, os.path.dirname(__file__))
from predict_rpe import convert_rpe_to_standard

print('='*60)
print('  GB + 配置难度外推（多押压制版）')
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

# split + 评估
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

# 全量训练
scaler_full = StandardScaler()
X_full = scaler_full.fit_transform(X)
gb_full = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
gb_full.fit(X_full, y)
print(f'\n  全量训练完成')

# ====== 外推Boost公式（重设计） ======
# 权重设计基于AT内部真实相关系数
# 密度突变(+0.43) > 变速(+0.42) > 密度/手速(+0.39) > 大跳(+0.39) > 多押(+0.35)
# 多押的cap从0.5降到0.10

def compute_boost(feats, p95):
    """配置难度外推 + 卡手/键盘区分
    
    键盘特征(多押+密度+手速) → 键盘谱多但这些pattern可预测 → 按顺手度扣减
    卡手特征(Hold锁手+重叠+位移突变) → 卡手=真正难 → 始终全额
    
    Rrhar'il(卡手): hold_lock=144/1300=0.11, hold_overlap=0.72, dt_mean=0.75
    QZKago(键盘): mf=86/1723=0.05, mf_eps=0.64, speed_change=148775
    """
    total_n = max(feats.get('total_notes', 1), 1)
    
    # ---- 键盘指数（温和扣减，不让键盘特征完全被smooth压死） ----
    mf_ratio = feats.get('multi_finger_3plus_events', 0) / total_n * 20
    mf_eps = feats.get('mf_events_per_second', 0) * 3
    speed_n = min(feats.get('speed_change_total_impact', 0) / 50000.0, 3.0)
    keyboard_idx = mf_ratio + mf_eps + speed_n
    
    # ---- 卡手指数 ----
    hold_lock_r = feats.get('hold_lock_tap_events', 0) / total_n * 10
    hold_ov = feats.get('hold_tap_overlap_ratio', 0) * 3
    dt_mean = feats.get('density_transition_mean', 0) / 0.65 * 1.5
    burst_mv = feats.get('burst_avg_movement', 0) / 5.0
    awkward_idx = hold_lock_r + hold_ov + dt_mean + burst_mv
    
    # ---- 顺滑度乘数（键盘越多越扣） ----
    smooth = (awkward_idx + 1.0) / (awkward_idx + keyboard_idx + 1.0)
    smooth = float(np.clip(smooth, 0.40, 0.85))
    
    # ===== 键盘特征boost（须经smooth扣减） =====
    kb_baselines = [
        ('multi_finger_3plus_events', 30.0, 0.09),
        ('wide_jump_count',          250.0, 0.09),
        ('micro_max_0.0625beat',     4.0, 0.12),
        ('notes_per_second',         10.0, 0.05),
        ('tap_per_second',           5.5, 0.04),
    ]
    kb_boost = 0.0
    for fname, baseline, coeff in kb_baselines:
        val = feats.get(fname, 0)
        p95_val = max(p95.get(fname, 0), baseline)
        if val > p95_val:
            kb_boost += coeff * float(np.log1p(val / p95_val - 1))
    kb_boost *= smooth
    
    # ===== 卡手特征boost（始终全额） =====
    ak_baselines = [
        ('density_transition_max',   4.0, 0.33),
        ('max_concurrent_holds',     3.0, 0.06),
        ('hold_lock_displacement_per_sec', 3.0, 0.13),
        ('hold_tap_overlap_ratio',   0.6, 0.12),
    ]
    ak_boost = 0.0
    for fname, baseline, coeff in ak_baselines:
        val = feats.get(fname, 0)
        p95_val = max(p95.get(fname, 0), baseline)
        if val > p95_val:
            ak_boost += coeff * float(np.log1p(val / p95_val - 1))
    
    # ===== 顺滑负向修正（节奏规整→负数） =====
    dom_r = feats.get('dominant_rhythm_ratio', 0.3)
    rh_ent = feats.get('rhythm_entropy', 2.5)
    smooth_penalty = 0.0
    if dom_r > 0.38:
        smooth_penalty -= (dom_r - 0.38) * 0.5
    if rh_ent < 2.2:
        smooth_penalty -= (2.2 - rh_ent) * 0.3
    smooth_penalty = max(smooth_penalty, -0.30)
    
    return min(kb_boost + ak_boost + smooth_penalty, 1.5)

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
    
    meta = f' ({raw["META"]["level"]})' if is_rpe else ''
    print(f'\n  {name}{meta}:')
    print(f'    GB={p_gb:.2f} + Boost={p_boost:.3f} = {p_final:.2f}')
    for k in ['hand_speed_index','multi_finger_3plus_events','hold_lock_tap_events',
              'hold_lock_displacement_per_sec','hold_tap_overlap_ratio',
              'density_transition_mean','burst_avg_movement','mf_events_per_second',
              'speed_change_total_impact']:
        print(f'    {k}: {feats.get(k,0):.2f} (P95={p95_vals.get(k,0):.2f})')
    results[name] = {'gb': p_gb, 'boost': p_boost, 'final': p_final}

# ====== 高难谱面评估 ======
print('\n' + '='*60)
print('  所有难度≥16 谱面 Boost 评估')
print('='*60)

high_items = []
for i, item in enumerate(all_items):
    if item['difficulty'] < 16.0: continue
    try:
        cd = load_chart_json(item['filepath'])
        feats = extract_features(cd)
        if not feats: continue
        x = np.array([[feats.get(n,0) for n in feature_names]])
        xs = scaler_full.transform(x)
        p_gb = float(gb_full.predict(xs)[0])
        p_boost = compute_boost(feats, p95_vals)
        p_final = p_gb + p_boost
        err = p_final - item['difficulty']
        high_items.append({
            'name': item['folder'], 'level': item['level'],
            'true': item['difficulty'], 'gb': p_gb, 'boost': p_boost,
            'pred': p_final, 'err': err
        })
    except: pass
    if (i+1)%300==0: print(f'  ...{i+1}/{len(all_items)}')

high_items.sort(key=lambda x: (-x['true'], x['err']))

for lv in ['IN', 'AT']:
    items = [r for r in high_items if r['level'] == lv]
    if not items: continue
    abs_errs = [abs(r['err']) for r in items]
    errs = [r['err'] for r in items]
    n = len(items)
    print(f'\n  {lv} (n={n}):')
    print(f'    平均真值: {np.mean([r["true"] for r in items]):.2f}')
    print(f'    平均预测: {np.mean([r["pred"] for r in items]):.2f}')
    print(f'    MAE: {np.mean(abs_errs):.3f}')
    print(f'    偏差: {np.mean(errs):+.3f}')
    print(f'    ±0.3以内: {sum(1 for e in abs_errs if e<=0.3)/n*100:.0f}%')
    print(f'    ±0.5以内: {sum(1 for e in abs_errs if e<=0.5)/n*100:.0f}%')
    print(f'    ±1.0以内: {sum(1 for e in abs_errs if e<=1.0)/n*100:.0f}%')

print(f'\n  偏差最大的10个:')
worst = sorted(high_items, key=lambda x: -abs(x['err']))[:10]
for r in worst:
    print(f'    {r["name"]:25s} {r["level"]:4s} 真={r["true"]:.1f} 预测={r["pred"]:.2f} 误差={r["err"]:+.3f} (boost={r["boost"]:.3f})')

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
print(f'  GB: R²={r2_gb:.4f}, MAE={mae_gb:.4f}')
for name, r in results.items():
    print(f'  {name}: {r["final"]:.2f} (GB={r["gb"]:.2f}, boost={r["boost"]:.3f})')
print(f'\n  保存: {save_path}')
