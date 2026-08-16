# -*- coding: utf-8 -*-
"""v13 训练: 官谱982 + 社区共识17+定数表(manifest17plus.tsv) 联合重训
- 特征/GB结构/P95修正与v12完全一致 (GB_KEEP/EXCLUDE不变)
- 社区共识表谱: 本地json匹配(正id缺的由fetch_manifest_charts.py下载), level=AT, 权重1.0
- 负id(已下架)无谱面文件, 跳过; Aegleseeker.pez仅测试不入训练
- CV只在官谱上分组评估; 社区共识表谱全量入训练, 另报in-sample MAE
用法: python train/train_v13.py --out 6dim_model_v13.pkl
"""
import os, sys, pickle, numpy as np, argparse, json
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from boost_config import MANUAL_FLAT

parser = argparse.ArgumentParser()
parser.add_argument('--atw', type=float, default=1.0, help='AT/16+段样本权重')
parser.add_argument('--loww', type=float, default=1.5, help='EZ/HD权重')
parser.add_argument('--w17', type=float, default=1.0, help='社区共识表17+谱权重')
parser.add_argument('--holdout', type=int, default=50, help='社区共识表留出测试数(seed42, 不入训练)')
parser.add_argument('--out', default='6dim_model_v13.pkl')
parser.add_argument('--caps', type=float, default=4.0)
parser.add_argument('--flat', default=None, help='JSON权重文件(opt_v13_boost输出), 覆盖MANUAL_FLAT')
parser.add_argument('--nest', type=int, default=500, help='GB树数')
parser.add_argument('--msl', type=int, default=3, help='min_samples_leaf')
parser.add_argument('--depth', type=int, default=5, help='max_depth')
parser.add_argument('--lr', type=float, default=0.05, help='learning_rate')
args = parser.parse_args()

FLAT = list(MANUAL_FLAT)
if args.flat and os.path.exists(args.flat):
    with open(args.flat, encoding='utf-8') as f:
        FLAT = [(a, b, c) for a, b, c in json.load(f)]
    print(f'使用优化权重: {args.flat} ({len(FLAT)}项)')
CAPS = {'_default': args.caps} if args.caps else {}
print(f'v13训练: atw={args.atw} loww={args.loww} w17={args.w17}')

# ===== 官谱 =====
CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)
all_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    for lv in ['EZ', 'HD', 'IN', 'AT']:
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
n_off = len(feats_list)
print(f'官谱特征成功: {n_off}')

# ===== 社区共识表 17+ =====
from unified_parser import load_chart_from_bytes
manifest = []
for l in open(os.path.join(_ROOT, 'data', 'phira', 'manifest17plus.tsv'), encoding='utf-8', errors='replace').read().splitlines()[2:]:
    p = l.split('	')
    if len(p) >= 3 and p[0].lstrip('-').isdigit():
        manifest.append((int(p[0]), p[1], float(p[2])))
m17_feats, m17_labels = [], []
m17_used, m17_skip = [], []
for cid, name, diff in manifest:
    if cid < 0:
        m17_skip.append((cid, name, '负id下架无谱面'))
        continue
    path = None
    for d in ['json_unranked_4star', 'json_unranked', 'json']:
        p = os.path.join(_ROOT, 'data', 'phira', d, '%d.json' % cid)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        m17_skip.append((cid, name, '本地无谱面'))
        continue
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        cd, pe = load_chart_from_bytes(raw)
        if cd is None:
            m17_skip.append((cid, name, '解析失败'))
            continue
        fe = extract_features(cd)
        if not fe:
            m17_skip.append((cid, name, '特征失败'))
            continue
        m17_feats.append(fe)
        m17_labels.append(diff)
        m17_used.append((cid, name, diff))
    except Exception as e:
        m17_skip.append((cid, name, str(e)[:50]))
print(f'社区共识表可用: {len(m17_feats)} / {len(manifest)} (跳过{len(m17_skip)})')

# 留出: 社区共识表随机留 holdout 首做泛化测试 (不入训练)
rng = np.random.RandomState(42)
m17_idx = np.arange(len(m17_feats))
hold_idx = rng.choice(m17_idx, size=min(args.holdout, len(m17_feats)), replace=False)
hold_set = set(hold_idx.tolist())
m17_train_idx = [i for i in m17_idx if i not in hold_set]
hold_feats = [m17_feats[i] for i in hold_idx]
hold_labels = [m17_labels[i] for i in hold_idx]
hold_names = [(m17_used[i][0], m17_used[i][1]) for i in hold_idx]

feats_list += [m17_feats[i] for i in m17_train_idx]
labels += [m17_labels[i] for i in m17_train_idx]
levels_list += ['AT'] * len(m17_train_idx)
names_list += ['m17#%d' % m17_used[i][0] for i in m17_train_idx]
n = len(feats_list)
print(f'总样本: {n} (官谱{n_off} + 社区共识表训练{len(m17_train_idx)} + 留出{len(hold_idx)})')

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
    'note_speed', 'flash_', 'visible_time', 'chord_jack', 'fast_hold',
]
GB_KEEP = {'density_dimension', 'real_core_notes_per_second',
           'core_peak_density_1sec_top5avg', 'core_peak_density_top5avg_1beat',
           'movement_per_second', 'movement_density_index',
           'jline_move_disp_per_sec', 'jline_rotate_disp_per_sec', 'jline_hidden_time_ratio',
           'hold_lock_weighted_per_sec', 'hold_lock_weighted_per_hold',
           # v13实验: 多指/配置特征重新进GB (社区共识表237首提供17+分布, 静态暴力多指TotalEclipse类欠定价)
           'multi_finger_3plus_events', 'multi_finger_4plus_events', 'multi_finger_max',
           'weighted_mf_score_per_sec', 'chord_alternation_rate', 'chord_size_entropy',
           'odd_division_ratio', 'type_switch_per_sec',
           'chord_3note', 'chord_4plus'}
gb_feature_names = [nn for nn in feature_names
                    if nn in GB_KEEP or not any(kw in nn for kw in GB_EXCLUDE_KEYWORDS)]
print(f'GB特征数: {len(gb_feature_names)}')

_JLINE_P95_FIX = {'jline_movement_density': 107.1, 'jline_rotate_density': 18.6, 'jline_disappear_density': 15.1}
def _apply_jline_p95_fix(p95_vals):
    for k, v in _JLINE_P95_FIX.items():
        if k in p95_vals:
            p95_vals[k] = v
    return p95_vals

X_base = np.array([[f.get(nn, 0) for nn in gb_feature_names] for f in feats_list])
y = np.array(labels)
LV_ORDER = ['EZ', 'HD', 'IN_AT']
X_lv = np.zeros((n, len(LV_ORDER)))
for i, lv in enumerate(levels_list):
    key = 'IN_AT' if lv in ('IN', 'AT') else lv
    X_lv[i, LV_ORDER.index(key)] = 1.0
levels_arr = np.array(levels_list)
SAMPLE_W = np.ones(n)
SAMPLE_W[levels_arr == 'EZ'] = args.loww
SAMPLE_W[levels_arr == 'HD'] = args.loww
hi_mask = (levels_arr == 'AT') | ((levels_arr == 'IN') & (y >= 16.0))
SAMPLE_W[hi_mask] = args.atw
m17_mask = np.zeros(n, dtype=bool)
m17_mask[n_off:] = True
SAMPLE_W[m17_mask] = args.w17
print(f'加权: EZ/HD={args.loww}, AT/IN16+={args.atw}, 社区共识表17+={args.w17}')

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
        if c is not None and e > c: e = c
        x = co * (e ** 0.70)
        if v > max(p99_vals.get(fname, 0), bl * 0.5):
            pe = v / max(p99_vals.get(fname, 0), bl * 0.5) - 1.0
            if c is not None and pe > c: pe = c
            x += co * max(0, pe) ** 0.70 * 0.5
        total += x
    return total

# CV: 官谱分组 + 社区共识表全量入训练(不进测试集)
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_base[:n_off], y[:n_off], groups=np.array(names_list[:n_off])))
oof_off = np.zeros(n_off)
oof_m17 = np.zeros(len(m17_train_idx))
for fi, (tr, te) in enumerate(splits):
    tr_full = np.concatenate([tr, np.arange(n_off, n)])
    p95_vals, p99_vals = {}, {}
    for j, name in enumerate(gb_feature_names):
        col = X_base[tr_full, j]
        p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
        p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
    _apply_jline_p95_fix(p95_vals)
    boosts = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in feats_list])
    X_tr = np.hstack([X_base[tr_full], X_lv[tr_full]])
    sc = StandardScaler().fit(X_tr)
    gb = GradientBoostingRegressor(n_estimators=args.nest, max_depth=args.depth, min_samples_leaf=args.msl,
                                   learning_rate=args.lr, subsample=0.8, random_state=42)
    gb.fit(sc.transform(X_tr), y[tr_full] - boosts[tr_full], sample_weight=SAMPLE_W[tr_full])
    oof_off[te] = gb.predict(sc.transform(np.hstack([X_base[te], X_lv[te]]))) + boosts[te]
    oof_m17 += gb.predict(sc.transform(np.hstack([X_base[n_off:], X_lv[n_off:]]))) + boosts[n_off:]
    print(f'  fold{fi}: 官谱OOF MAE={mean_absolute_error(y[:n_off][te], oof_off[te]):.4f}', flush=True)
oof_m17 /= len(splits)

errs_off = oof_off - y[:n_off]
print(f'\n===== 官谱歌曲分组CV (n={n_off}) =====')
print(f'整体MAE = {mean_absolute_error(y[:n_off], oof_off):.4f} | bias = {errs_off.mean():+.4f}')
for lv in ['EZ', 'HD', 'IN', 'AT']:
    mk = np.where(levels_arr[:n_off] == lv)[0]
    print(f'  {lv}: n={len(mk)} MAE={mean_absolute_error(y[mk], oof_off[mk]):.4f} bias={errs_off[mk].mean():+.4f}')
m17_tr_labels = [m17_labels[i] for i in m17_train_idx]
errs_m17 = oof_m17 - np.array(m17_tr_labels)
print(f'\n===== 社区共识表17+ (OOF, n={len(m17_train_idx)}) =====')
print(f'MAE = {mean_absolute_error(m17_tr_labels, oof_m17):.4f} | bias = {errs_m17.mean():+.4f}')
for lo, hi, tag in [(17, 17.5, '17-17.5'), (17.5, 18, '17.5-18'), (18, 18.5, '18-18.5'), (18.5, 19, '18.5-19'), (19, 99, '19+')]:
    mk = np.where((np.array(m17_tr_labels) >= lo) & (np.array(m17_tr_labels) < hi))[0]
    if len(mk):
        print(f'  [{tag}]: n={len(mk)} MAE={mean_absolute_error(np.array(m17_tr_labels)[mk], oof_m17[mk]):.4f} bias={errs_m17[mk].mean():+.4f}')

# 最终模型: 全量拟合
p95_vals, p99_vals = {}, {}
for j, name in enumerate(gb_feature_names):
    col = X_base[:, j]
    p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
    p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
_apply_jline_p95_fix(p95_vals)
boosts = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in feats_list])
X_all = np.hstack([X_base, X_lv])
scaler = StandardScaler().fit(X_all)
gb_final = GradientBoostingRegressor(n_estimators=args.nest, max_depth=args.depth, min_samples_leaf=args.msl,
                                     learning_rate=args.lr, subsample=0.8, random_state=42)
gb_final.fit(scaler.transform(X_all), y - boosts, sample_weight=SAMPLE_W)
pred_all = gb_final.predict(scaler.transform(X_all)) + boosts
print(f'\n===== 最终模型 in-sample =====')
print(f'官谱 in-sample MAE = {mean_absolute_error(y[:n_off], pred_all[:n_off]):.4f} bias={np.mean(pred_all[:n_off]-y[:n_off]):+.4f}')
print(f'社区共识表 in-sample MAE = {mean_absolute_error(y[n_off:], pred_all[n_off:]):.4f} bias={np.mean(pred_all[n_off:]-y[n_off:]):+.4f}')

# 留出集预测
if hold_feats:
    X_hold = np.array([[f.get(nn, 0) for nn in gb_feature_names] for f in hold_feats])
    X_hold_lv = np.zeros((len(hold_feats), len(LV_ORDER)))
    X_hold_lv[:, LV_ORDER.index('IN_AT')] = 1.0
    boost_hold = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in hold_feats])
    pred_hold = gb_final.predict(scaler.transform(np.hstack([X_hold, X_hold_lv]))) + boost_hold
    eh = pred_hold - np.array(hold_labels)
    print(f'\n===== 社区共识表留出 (n={len(hold_idx)}) =====')
    print(f'MAE = {mean_absolute_error(hold_labels, pred_hold):.4f} | bias = {eh.mean():+.4f}')
    for i in range(len(hold_idx)):
        print(f'  #%-6d %-30s 表%.1f 预测%.2f 差%+.2f' % (hold_names[i][0], hold_names[i][1][:30], hold_labels[i], pred_hold[i], eh[i]))

out = os.path.join(_ROOT, 'models', args.out)
with open(out, 'wb') as f:
    pickle.dump({'gb': gb_final, 'scaler': scaler,
                 'feature_names': gb_feature_names,
                 'p95_vals': p95_vals, 'p99_vals': p99_vals,
                 'lv_order': LV_ORDER, 'caps': CAPS,
                 'MANUAL_FLAT': FLAT, 'n_train': n,
                 'm17_count': len(m17_train_idx), 'm17_used': m17_used,
                 'm17_skip': m17_skip, 'v13': True}, f)
print(f'已保存: {out}')
# 保存留出/OOF/特征供线性调参 (GB残差 = 预测 - 全局boost)
with open(os.path.join(_ROOT, 'models', 'v13_aux.pkl'), 'wb') as f:
    pickle.dump({'hold_names': hold_names, 'hold_labels': hold_labels,
                 'hold_feats': hold_feats, 'hold_gb': (pred_hold - boost_hold) if hold_feats else None,
                 'off_feats': feats_list[:n_off], 'oof_gb': oof_off - boosts[:n_off],
                 'y_off': y[:n_off], 'gb_feature_names': gb_feature_names}, f)
print('已保存: models/v13_aux.pkl')
