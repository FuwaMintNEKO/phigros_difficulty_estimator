# -*- coding: utf-8 -*-
"""v10 最终生产模型: 全量982官谱训练
  1. 5折CV生成OOF (与train_v10一致: GB残差+level特征) → 报告诚实CV MAE
  2. (isotonic按level校准已在诚实CV中被否决: 0.572 > 0.546, 不纳入)
  3. 用全部982谱重训最终GB
  4. 保存 models/6dim_model_v10.pkl: gb, scaler, FN, P95, P99, MANUAL_FLAT, caps, meta

  --variant 选择 boost 权重变体 (V14=差速+和弦重键 cap4 / V15=再加 tsw/alt 压缩)
  --caps 设置全局 excess 封顶 (默认4.0)
"""
import os, sys, pickle, numpy as np, argparse
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from boost_config import MANUAL_FLAT

parser = argparse.ArgumentParser()
parser.add_argument('--variant', default='V14', choices=['V14', 'V15'])
parser.add_argument('--caps', type=float, default=4.0)
parser.add_argument('--lowweight', type=float, default=1.5,
                    help='低段(EZ/HD)样本权重, 1.0=不加权 (分组CV实验: 1.5 最佳)')
parser.add_argument('--lvmerge', type=int, default=1,
                    help='合并IN/AT等级 (1=3类EZ/HD/IN_AT, 0=4类) — 实验已验证有效 MAE 0.5383→0.5324')
args = parser.parse_args()

# 变体权重覆盖 (对应 tools/boost_weight_experiment.py 的 V14/V15)
OVERRIDE = {
    'V14': {},
    'V15': {'type_switch_per_sec': 0.06, 'chord_alternation_rate': 0.15,
            'weighted_mf_score_per_sec': 0.20, 'above_avg_duration_sec': 0.42},
}[args.variant]
FLAT = []
d = {f: (bl, co) for f, bl, co in MANUAL_FLAT}
for f, new_co in OVERRIDE.items():
    d[f] = (d[f][0], new_co)
FLAT = [(f, bl, co) for f, (bl, co) in d.items()]
CAPS = {'_default': args.caps} if args.caps else {}
print(f'变体: {args.variant}, 全局cap: {args.caps}, boost特征数: {len(FLAT)}')

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)
all_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    for lv in ['EZ','HD','IN','AT']:
        if lv in info['levels'] and lv in diffs:
            all_items.append({'folder': fn, 'filepath': info['levels'][lv],
                              'difficulty': diffs[lv], 'level': lv})
print(f'官谱总数: {len(all_items)}')

feats_list, labels, levels_list, names_list = [], [], [], []
for item in all_items:
    try:
        cd = load_chart_json(item['filepath'])
        feats = extract_features(cd)
        if feats:
            feats_list.append(feats); labels.append(item['difficulty'])
            levels_list.append(item['level']); names_list.append(item['folder'])
    except Exception:
        pass
n = len(feats_list)
print(f'特征提取成功: {n}')

feature_names = sorted(feats_list[0].keys())
GB_EXCLUDE_KEYWORDS = [
    'stop_go', 'track_section', 'offbeat_ratio', 'dense_mf',
    'mf_burst', 'mf_events_per_second', 'mf_with_hold',
    'cross_line_3plus', 'min_interval_beats',
    'multi_finger_3plus', 'multi_finger_4plus', 'multi_finger_max',
    'chord_size_entropy', 'chord_3note', 'chord_4plus',
    'long_jack', 'short_jack', 'jack_max_run',
    'per_second', 'per_sec', 'rate_per_sec',
    'total_movement', 'total_steps', 'total_event',
    'total_hold_duration', 'total_chord',
    'speed_change_total',
    'micro_max_', 'micro_spike_',
    'density_above_zero', 'core_density_above_zero',
    'density_skew', 'density_transition_max',
    'avg_hold_duration', 'max_hold_duration',
    'finger_vs_total',
    # 差速/闪现/和弦重键: 训练集(官谱)无差速样本, GB 外推不可靠 → 仅由 boost 负责
    'note_speed', 'flash_', 'visible_time', 'chord_jack', 'fast_hold',
]
GB_KEEP = {'density_dimension', 'real_core_notes_per_second',
           'core_peak_density_1sec_top5avg', 'core_peak_density_top5avg_1beat',
           # 2026-08-13: 修复阈值后 movement 是有用特征, 豁免 'per_second' 排除规则
           'movement_per_second', 'movement_density_index'}
gb_feature_names = [nn for nn in feature_names
                    if nn in GB_KEEP or not any(kw in nn for kw in GB_EXCLUDE_KEYWORDS)]
print(f'GB特征数: {len(gb_feature_names)}')

X_base = np.array([[f.get(nn, 0) for nn in gb_feature_names] for f in feats_list])
y = np.array(labels)
LV_ORDER = ['EZ', 'HD', 'IN', 'AT']
if args.lvmerge:
    LV_ORDER = ['EZ', 'HD', 'IN_AT']  # IN 与 AT 合并 (验证有效: 定数连续, 不因标签突变)
X_lv = np.zeros((n, len(LV_ORDER)))
for i, lv in enumerate(levels_list):
    key = 'IN_AT' if (args.lvmerge and lv in ('IN', 'AT')) else lv
    X_lv[i, LV_ORDER.index(key)] = 1.0
levels_arr = np.array(levels_list)
# 低段样本加权 (分组CV实验: EZ/HD weight=1.5 最优, 与尾杀特征互补)
SAMPLE_W = np.ones(n)
if args.lowweight != 1.0:
    SAMPLE_W[levels_arr == 'EZ'] = args.lowweight
    SAMPLE_W[levels_arr == 'HD'] = args.lowweight
print(f'低段加权: EZ/HD weight={args.lowweight}')

def compute_boost_v9(feats, p95_vals, p99_vals):
    total = 0.0
    cap = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0)
        pv = p95_vals.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t: continue
        e = v / t - 1.0
        c = CAPS.get(fname, cap)
        if c is not None and e > c:
            e = c
        x = co * (e ** 0.70)
        if v > max(p99_vals.get(fname, 0), bl * 0.5):
            pe = v / max(p99_vals.get(fname, 0), bl * 0.5) - 1.0
            if c is not None and pe > c:
                pe = c
            x += co * max(0, pe) ** 0.70 * 0.5
        total += x
    return total

# ===== 1. 歌曲分组5折CV生成OOF (诚实泛化口径: 整首歌进测试集, 无同曲泄漏) =====
N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)
splits = list(gkf.split(X_base, y, groups=np.array(names_list)))
oof = np.zeros(n)
for fi, (tr, te) in enumerate(splits):
    p95_vals, p99_vals = {}, {}
    for j, name in enumerate(gb_feature_names):
        col = X_base[tr, j]
        p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
        p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
    boosts = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in feats_list])
    X_tr = np.hstack([X_base[tr], X_lv[tr]])
    X_te = np.hstack([X_base[te], X_lv[te]])
    sc = StandardScaler().fit(X_tr)
    gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
    gb.fit(sc.transform(X_tr), y[tr] - boosts[tr], sample_weight=SAMPLE_W[tr])
    oof[te] = gb.predict(sc.transform(X_te)) + boosts[te]
    print(f'  fold{fi} OOF完成', flush=True)

# ===== 2. 诚实CV评估 (歌曲分组CV, 即真实新谱泛化误差) =====
print(f'\n===== 歌曲分组5折CV评估 (OOF) =====')
print(f'整体MAE = {mean_absolute_error(y, oof):.4f}')
print(f'整体R2  = {r2_score(y, oof):.4f}')
for lv in LV_ORDER:
    m = np.where(levels_arr == lv)[0]
    if len(m):
        print(f'  {lv}: n={len(m)} MAE={mean_absolute_error(y[m], oof[m]):.4f}')
print('(isotonic按level校准在诚实CV实验中更差 0.572>0.546, 已否决, 不纳入最终模型)')

# ===== 3. 全量重训最终GB =====
p95_vals, p99_vals = {}, {}
for j, name in enumerate(gb_feature_names):
    col = X_base[:, j]
    p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
    p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
boosts = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in feats_list])
X_all = np.hstack([X_base, X_lv])
scaler = StandardScaler().fit(X_all)
gb_final = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
gb_final.fit(scaler.transform(X_all), y - boosts, sample_weight=SAMPLE_W)

# ===== 4. 保存 =====
model = {
    'gb': gb_final,
    'scaler': scaler,
    'feature_names': gb_feature_names,
    'p95_vals': p95_vals,
    'p99_vals': p99_vals,
    'lv_order': LV_ORDER,
    'version': f'10.1-{args.variant}-tail-low{args.lowweight}',
    'n_train': n,
    'MANUAL_FLAT': FLAT,
    'caps': CAPS,
    'train_meta': {'n': n, 'songs': len(set(names_list)), 'cv_mae': float(mean_absolute_error(y, oof)),
                   'cv_r2': float(r2_score(y, oof))},
}
path = os.path.join(_ROOT, 'models', '6dim_model_v10.pkl')
with open(path, 'wb') as f:
    pickle.dump(model, f)
print(f'\n已保存: {path}')
print(f'特征数: {len(gb_feature_names)}, 训练谱面: {n}')
