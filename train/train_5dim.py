import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, numpy as np
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
print('  Phigros 5维度难度预测系统 v2.1')
print('  统一模型: 不分IN/AT, 按定数统一预测 | 极保守boost')
print('='*70)

# ====== 1. 加载数据 ======
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
print(f'\n总样本: {len(all_items)}')

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

feature_names = sorted(feats_list[0].keys())
X_full = np.array([[f.get(n,0) for n in feature_names] for f in feats_list])
y_full = np.array(labels)
lv_arr = np.array(levels_list)
n_feats = len(feature_names)
n_samples = len(feats_list)
print(f'\n成功: {n_samples} 谱面, {n_feats} 特征, 难度 {y_full.min():.1f}~{y_full.max():.1f}')

# P95和P99
p95_vals, p99_vals = {}, {}
for j, name in enumerate(feature_names):
    col = X_full[:, j]
    p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
    p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0

# 特征与难度的相关系数
corrs_dict = {}
for j, name in enumerate(feature_names):
    col = X_full[:, j]
    if np.std(col) > 0 and np.std(y_full) > 0:
        corrs_dict[name] = float(np.corrcoef(col, y_full)[0, 1])
    else:
        corrs_dict[name] = 0

# ====== 2. 训练GB (全量数据,含AT) ======
# GB处理内推, 5维度boost处理极端外推(特征超P95)
print('\n--- 训练GB内推模型 (全量957张) ---')
mask_any = np.ones(len(y_full), dtype=bool)
X_train_gb = X_full[mask_any]
y_train_gb = y_full[mask_any]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
bins = np.digitize(y_train_gb, bins=[0,5,7,9,11,13,14,15,16,16.5,17])
train_idx, test_idx = next(sss.split(X_train_gb, bins))

scaler_gb = StandardScaler()
X_tr_s = scaler_gb.fit_transform(X_train_gb[train_idx])
X_te_s = scaler_gb.transform(X_train_gb[test_idx])
y_tr, y_te = y_train_gb[train_idx], y_train_gb[test_idx]

gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                learning_rate=0.05, subsample=0.8, random_state=42)
gb.fit(X_tr_s, y_tr)
y_pred_gb = gb.predict(X_te_s)
r2_gb = r2_score(y_te, y_pred_gb)
mae_gb = mean_absolute_error(y_te, y_pred_gb)
print(f'  测试集: R²={r2_gb:.4f}, MAE={mae_gb:.4f}')

# 全量训练
X_gb_s = scaler_gb.fit_transform(X_train_gb)
gb_full = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
gb_full.fit(X_gb_s, y_train_gb)
print(f'\n  全量训练完成 (n={len(y_train_gb)})')

# ====== 3. 五大维度外推boost ======
# 设计原则:
#   - GB训练在EZ+HD+IN上，对AT完全外推
#   - 每个维度独立计算，基于全量相关性+社区知识
#   - log1p缩放: 极端外推收益递减
#   - 读谱维度权重最高(全量r=0.87, AT内部更重要)
#   - 多押维度区分"可拆分"vs"强制多指"

def _dim_boost(feats, p99, feat_list, min_trig, div=2.0):
    """计算单个维度的boost值

    策略:
      - sqrt(raw) 压缩单一极端特征
      - trig_factor: 需要 min_trig 个特征同时触发才算数
      - div 进一步压低幅度, 因为GB已经预测得很准了
    """
    raw = 0.0
    trig_count = 0
    for fname, baseline, coeff in feat_list:
        val = feats.get(fname, 0)
        thresh = max(p99.get(fname, 0), baseline)
        if val > thresh:
            raw += coeff * float(np.log1p(val / thresh - 1))
            trig_count += 1
    if trig_count == 0:
        return 0.0, 0
    trig_factor = min(1.0, trig_count / max(min_trig, 1))
    return float(np.sqrt(raw)) * trig_factor / div, trig_count


def compute_5dim_boost(feats, p95, p99):
    """5维度非线性外推boost — 宽松版

    更新:
      - D1: 使用tap_micro_max(只算Tap蓝键), 降低min_trig和div
      - D4: 使用tap_notes_per_second(排除Drag/Flick)
      - D5: 新增type_switch_ratio(红蓝黄交替频率)
      - 提高cap从0.30→0.50, BPM变化对手速谱影响更明显
    """
    total_n = max(feats.get('total_notes', 1), 1)

    # ====== 维度1: 交互/纵连 (手速) ======
    # 使用tap_micro_max(只算Tap蓝键)替代micro_max(含Drag/Flick)
    dim1_feats = [
        ('tap_micro_max_0.0625beat',  2.0,  0.85),
        ('tap_burst_top5',             8.0,  0.55),
        ('jack_count',                35.0,  0.40),
        ('tap_per_second',             5.0,  0.40),
        ('very_short_interval_ratio',  0.25, 0.35),
    ]
    dim1, trig1 = _dim_boost(feats, p99, dim1_feats, 2, div=2.0)

    # ====== 维度2: 多押/多指 (仅强制多指) ======
    mf3 = feats.get('multi_finger_3plus_events', 0)
    spread_max = feats.get('sim_pos_spread_max', 0)
    spread_mean = feats.get('sim_pos_spread_mean', 0.5)

    forced_mf_idx = mf3 * spread_max / max(total_n, 1) * 10
    splittable_mf_idx = mf3 * max(1.0 - spread_mean, 0) / max(total_n, 1) * 5

    dim2 = 0.0
    trig2 = 0
    thresh_fmf = max(p99.get('multi_finger_3plus_events', 30), 1) * max(p99.get('sim_pos_spread_max', 0.8), 0.1) / max(p99.get('total_notes', 500), 1) * 10
    if forced_mf_idx > max(thresh_fmf, 0.8):
        dim2 = float(np.sqrt(max(float(np.log1p(forced_mf_idx / max(thresh_fmf, 0.8) - 1)), 0))) / 1.5
        trig2 += 1

    if splittable_mf_idx > 1.0:
        dim2 -= 0.08 * min(float(np.log1p(splittable_mf_idx)), 1.0)
    dim2 = max(dim2, -0.05)

    # ====== 维度3: 位移 ======
    dim3_feats = [
        ('wide_jump_count',            120.0, 0.50),
        ('burst_avg_movement',           2.5, 0.40),
        ('hold_lock_displacement_per_sec', 1.5, 0.50),
        ('hold_tap_overlap_ratio',      0.4,  0.25),
    ]
    dim3, trig3 = _dim_boost(feats, p99, dim3_feats, 2, div=1.8)

    # ====== 维度4: 耐力 ======
    # 使用tap_notes_per_second排除Drag/Flick充数
    dim4_feats = [
        ('total_notes',               1100.0, 0.55),
        ('tap_notes_per_second',        7.5,  0.40),
        ('high_density_duration_ratio_16beat', 0.30, 0.25),
        ('std_density_1beat',          0.25,  0.18),
    ]
    dim4, trig4 = _dim_boost(feats, p99, dim4_feats, 2, div=1.8)
    dim4 = min(dim4, 0.70)

    # ====== 维度5: 读谱 ======
    dim5_feats = [
        ('density_transition_max',      4.0,  0.90),
        ('tempo_change_count',         60.0,  0.70),
        ('speed_change_total_impact', 60000,  0.35),
        ('offbeat_ratio',              0.20,  0.40),
        ('rhythm_entropy',             4.5,   0.22),
        ('bpm_change_count',            2.0,  0.40),
        ('density_transition_mean',     0.55, 0.45),
        ('type_switch_ratio',           0.15, 0.30),
    ]
    dim5, trig5 = _dim_boost(feats, p99, dim5_feats, 2, div=1.8)

    # ====== 非线性汇总 ======
    raw_total = dim1 * 0.15 + dim2 * 0.08 + dim3 * 0.15 + dim4 * 0.15 + dim5 * 0.28

    cap = 0.50
    total_boost = cap * float(np.tanh(raw_total / cap))
    total_boost = min(total_boost, 0.80)

    return total_boost, {'dim1_交互纵连': round(dim1, 3), 'dim2_多押': round(dim2, 3),
                          'dim3_位移': round(dim3, 3), 'dim4_耐力': round(dim4, 3),
                          'dim5_读谱': round(dim5, 3),
                          'triggers': f'{trig1}/{trig2}/{trig3}/{trig4}/{trig5}'}

# ====== 4. 测试谱面 ======
print('\n' + '='*70)
print('  参考谱面评估')
print('='*70)

test_charts = [
    ('Chart_SP',           os.path.join(_ROOT, 'data', 'chart', 'Chart_SP.json'), False),
    ('Chart_SP #13',       os.path.join(_ROOT, 'data', 'chart', 'Chart_SP #1347(1).json'), False),
    ('Regrets',            os.path.join(_ROOT, 'data', 'chart', 'Sigma (Haocore Mix) ~ Regrets of The Yellow Tuli.json'), False),
    ('105秒伝說',          os.path.join(_ROOT, 'data', 'chart', 'Sigma (Haocore Mix) ~ 105秒の伝說 ~.json'), False),
    ('Aether Crest',       os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json'), True),
]

results = {}
for name, path, is_rpe in test_charts:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    cd = convert_rpe_to_standard(raw) if is_rpe else raw
    feats = extract_features(cd)
    if not feats: continue

    x = np.array([[feats.get(n,0) for n in feature_names]])
    xs = scaler_gb.transform(x)
    p_gb = float(gb_full.predict(xs)[0])
    p_boost, dims = compute_5dim_boost(feats, p95_vals, p99_vals)
    p_final = p_gb + p_boost

    meta = f' ({raw["META"]["level"]})' if is_rpe else ''
    print(f'\n  {name}{meta}:')
    print(f'    GB={p_gb:.2f} + Boost={p_boost:.3f} = {p_final:.2f}')
    for k, v in dims.items():
        print(f'    {k}: {v}')
    results[name] = {'gb': p_gb, 'boost': p_boost, 'final': p_final, 'dims': dims}

# ====== 5. 统一难度评估 (不分IN/AT) ======
print('\n' + '='*70)
print('  统一难度评估 — 按定数分组 (不分IN/AT)')
print('='*70)

high_items = []
all_boosted = []
for i, item in enumerate(all_items):
    try:
        cd = load_chart_json(item['filepath'])
        feats = extract_features(cd)
        if not feats: continue
        x = np.array([[feats.get(n,0) for n in feature_names]])
        xs = scaler_gb.transform(x)
        p_gb = float(gb_full.predict(xs)[0])
        p_boost, dims = compute_5dim_boost(feats, p95_vals, p99_vals)
        p_final = p_gb + p_boost
        err = p_final - item['difficulty']
        rec = {
            'name': item['folder'], 'level': item['level'], 'diff': item['difficulty'],
            'true': item['difficulty'], 'gb': p_gb, 'boost': p_boost,
            'pred': p_final, 'err': err, 'dims': dims,
            'feats': feats,
        }
        all_boosted.append(rec)
        if item['difficulty'] >= 15.5:
            high_items.append(rec)
    except: pass

# 按难度区间分组 (不分IN/AT)
diff_buckets = [(15.5, 16.0), (16.0, 16.5), (16.5, 17.0), (17.0, 18.0)]
for lo, hi in diff_buckets:
    items = [r for r in high_items if lo <= r['true'] < hi]
    if not items: continue
    abs_errs = [abs(r['err']) for r in items]
    errs = [r['err'] for r in items]
    n = len(items)
    levels = ','.join(sorted(set(r['level'] for r in items)))
    print(f'\n  难度 {lo:.1f}~{hi:.1f} (n={n}, {levels}):')
    print(f'    平均真值: {np.mean([r["true"] for r in items]):.2f}')
    print(f'    平均预测: {np.mean([r["pred"] for r in items]):.2f}')
    print(f'    MAE: {np.mean(abs_errs):.3f}')
    print(f'    偏差: {np.mean(errs):+.3f}')
    print(f'    ±0.1以内: {sum(1 for e in abs_errs if e<=0.1)/n*100:.0f}%')
    print(f'    ±0.3以内: {sum(1 for e in abs_errs if e<=0.3)/n*100:.0f}%')

# 偏差最大的10个 (全量)
print(f'\n  偏差最大的10个:')
worst = sorted(high_items, key=lambda x: -abs(x['err']))[:10]
for r in worst:
    print(f'  {r["name"][:30]:30s} {r["level"]:4s} 真={r["true"]:.1f} 预测={r["pred"]:.2f} 误差={r["err"]:+.3f} (boost={r["boost"]:.3f})')

print(f'\n  全部高难(n={len(high_items)}): MAE={np.mean([abs(r["err"]) for r in high_items]):.3f}, '
      f'偏差={np.mean([r["err"] for r in high_items]):+.3f}')

# ====== 5b. 参考谱面详细5维分解 ======

# ====== 6. 找用户提到的参考谱面 ======
print('\n' + '='*70)
print('  社区参考谱面查找 + 详细5维分解')
print('='*70)

ref_patterns = [
    ('Destruction 3,2,1 (手速谱)', 'DESTRUCTION', None),
    ('QZKago (手速谱/键盘谱)', 'QZKago', None),
    ('Rrhar\'il AT (atrr, 卡手谱)', 'Rrhar', 'AT'),
    ('Stardust:RAY (AT17.2键盘谱)', 'Stardust', 'AT'),
    ('+ERABY+E (AT17.3键盘谱)', 'ERABY', 'AT'),
    ('Distorted Fate (AT17.4)', 'DistortedFate', 'AT'),
    ('Alice in a xxxxx (读谱谱)', 'Alice in a', None),
    ('玩具狂奏曲 (读谱谱)', '玩具狂奏', None),
    ('Pragmatism (读谱谱)', 'Pragmatism', None),
    ('Slips (读谱谱)', 'Slips', None),
]

for label, pattern, force_lv in ref_patterns:
    candidates = []
    for r in all_boosted:
        name_lower = r['name'].lower()
        if pattern.lower() in name_lower:
            if force_lv is None or r['level'] == force_lv:
                # prefer IN/AT over lower diffs
                candidates.append(r)
    if not candidates:
        continue
    # Show only the hardest 2 levels
    seen_names = set()
    for r in sorted(candidates, key=lambda x: -x['true'])[:3]:
        if r['name'] in seen_names: continue
        seen_names.add(r['name'])
        print(f'\n  [{label}] {r["name"][:35]:35s} {r["level"]:4s} 真={r["true"]:.1f}')
        print(f'    GB={r["gb"]:.2f} + Boost={r["boost"]:.3f} = {r["pred"]:.2f} (误差={r["err"]:+.3f})')
        for dk, dv in r['dims'].items():
            if isinstance(dv, str):
                print(f'    {dk}: {dv}')
                continue
            bar = '█' * max(0, min(30, int((dv + 0.3) * 15)))
            print(f'    {dk}: {dv:+.3f} {bar}')
        for key in ['notes_per_second', 'tap_per_second', 'jack_count',
                    'multi_finger_3plus_events', 'burst_avg_movement',
                    'wide_jump_count', 'hold_lock_displacement_per_sec',
                    'density_transition_max', 'tempo_change_count',
                    'speed_change_total_impact', 'offbeat_ratio',
                    'rhythm_entropy', 'bpm_range', 'bpm_change_count',
                    'total_notes']:
            v = r['feats'].get(key, 0)
            p99_v = p99_vals.get(key, 0)
            flag = ' ↑↑' if v > p99_v else (' ↑' if v > p99_v * 0.85 else '')
            print(f'      {key:35s} = {v:8.2f} (P99={p99_v:7.2f}){flag}')

# ====== 7. 保存 ======
model_data = {
    'gb': gb_full, 'scaler': scaler_gb, 'feature_names': feature_names,
    'p95_vals': p95_vals, 'p99_vals': p99_vals, 'corrs_dict': corrs_dict,
    'gb_metrics': {'r2': r2_gb, 'mae': mae_gb},
}
save_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, '5dim_model.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(model_data, f)

# 保存全量预测结果
with open(os.path.join(save_dir, '5dim_predictions.csv'), 'w', encoding='utf-8') as f:
    f.write('name,level,true,gb,boost,pred,err,'
            + 'dim1_交互纵连,dim2_多押,dim3_位移,dim4_耐力,dim5_读谱\n')
    for r in all_boosted:
        f.write(f'{r["name"]},{r["level"]},{r["true"]:.1f},{r["gb"]:.3f},{r["boost"]:.3f},'
                f'{r["pred"]:.3f},{r["err"]:.3f},'
                f'{r["dims"]["dim1_交互纵连"]},{r["dims"]["dim2_多押"]},'
                f'{r["dims"]["dim3_位移"]},{r["dims"]["dim4_耐力"]},{r["dims"]["dim5_读谱"]}\n')

print('\n' + '='*70)
at_items = [r for r in all_boosted if r['level'] == 'AT']
if at_items:
    print(f'  AT (n={len(at_items)}): MAE={np.mean([abs(r["err"]) for r in at_items]):.3f}, '
          f'偏差={np.mean([r["err"] for r in at_items]):+.3f}')
print(f'  全部高难(n={len(high_items)}): MAE={np.mean([abs(r["err"]) for r in high_items]):.3f}, '
      f'偏差={np.mean([r["err"] for r in high_items]):+.3f}')
print(f'  保存: {save_path}')
print('='*70)
