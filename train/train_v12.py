# -*- coding: utf-8 -*-
"""v12 训练: 在 v11.13 基线(train_v11_a.py atw=1.0 loww=1.5 boostvar=none)上,
使用 v11.15e 彻查修复后的特征(单位/阈值/分音/判定线位移量等 30+ 处修正)重训。
与 v11.13 唯一差异: 输入特征修复 + 新增3个跨格式可比jline特征入GB。
用法: python train/train_v12.py --out 6dim_model_v12.pkl
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
parser.add_argument('--atw', type=float, default=1.0, help='AT/16+段样本权重 (v11.13基线=1.0)')
parser.add_argument('--loww', type=float, default=1.5, help='EZ/HD权重 (v11.13基线=1.5)')
parser.add_argument('--out', default='6dim_model_v12.pkl')
parser.add_argument('--caps', type=float, default=4.0)
args = parser.parse_args()

# v11.13基线: MANUAL_FLAT 原样使用 boost_config (手工boost层不变)
FLAT = list(MANUAL_FLAT)
CAPS = {'_default': args.caps} if args.caps else {}
print(f'v12训练: atw={args.atw} loww={args.loww} boost特征数={len(FLAT)}')

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
    'note_speed', 'flash_', 'visible_time', 'chord_jack', 'fast_hold',
]
GB_KEEP = {'density_dimension', 'real_core_notes_per_second',
           'core_peak_density_1sec_top5avg', 'core_peak_density_top5avg_1beat',
           'movement_per_second', 'movement_density_index',
           # v12: 新增跨格式可比的判定线位移特征 (修复前RPE谱jline事件完全丢失)
           'jline_move_disp_per_sec', 'jline_rotate_disp_per_sec', 'jline_hidden_time_ratio',
           # v12.5: 锁手加权(含连续接条) — 全长条谱的锁手维度此前被per_sec关键字误排除
           'hold_lock_weighted_per_sec', 'hold_lock_weighted_per_hold'}
gb_feature_names = [nn for nn in feature_names
                    if nn in GB_KEEP or not any(kw in nn for kw in GB_EXCLUDE_KEYWORDS)]
print(f'GB特征数: {len(gb_feature_names)} (含新jline位移3特征)')

# v12.1: 训练与app.py推理必须用同一P95 — 同步app.py的jline P95修正
# (训练数据P95=301/158/207被瞬移演出谱污染, app修正为107.1/18.6/15.1;
#  不同步会导致网页端boost系统性偏大, 官谱预测整体+0.11)
_JLINE_P95_FIX = {'jline_movement_density': 107.1, 'jline_rotate_density': 18.6, 'jline_disappear_density': 15.1}

def _apply_jline_p95_fix(p95_vals):
    for k, v in _JLINE_P95_FIX.items():
        if k in p95_vals:
            p95_vals[k] = v
    return p95_vals
print(f'特征总数: {len(feature_names)} (v11.15e删has_AT, 加3个jline位移特征)')

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
print(f'加权: EZ/HD={args.loww}, AT/IN16+={args.atw} (n_hi={hi_mask.sum()})')

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

gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X_base, y, groups=np.array(names_list)))
oof = np.zeros(n)
for fi, (tr, te) in enumerate(splits):
    p95_vals, p99_vals = {}, {}
    for j, name in enumerate(gb_feature_names):
        col = X_base[tr, j]
        p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
        p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
    _apply_jline_p95_fix(p95_vals)  # v12.1: 与app.py推理P95一致
    boosts = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in feats_list])
    X_tr = np.hstack([X_base[tr], X_lv[tr]])
    X_te = np.hstack([X_base[te], X_lv[te]])
    sc = StandardScaler().fit(X_tr)
    gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
    gb.fit(sc.transform(X_tr), y[tr] - boosts[tr], sample_weight=SAMPLE_W[tr])
    oof[te] = gb.predict(sc.transform(X_te)) + boosts[te]
    print(f'  fold{fi} OOF完成', flush=True)

errs = oof - y
print(f'\n===== 官谱歌曲分组CV =====')
print(f'整体MAE = {mean_absolute_error(y, oof):.4f} (v11.13基线=0.5199) | 整体bias = {errs.mean():+.4f}')
for lv in ['EZ', 'HD', 'IN', 'AT']:
    mk = np.where(levels_arr == lv)[0]
    print(f'  {lv}: n={len(mk)} MAE={mean_absolute_error(y[mk], oof[mk]):.4f} bias={errs[mk].mean():+.4f}')
for lo, hi, tag in [(11, 13, '11-13'), (13, 14, '13-14'), (14, 15, '14-15'), (15, 16, '15-16'), (16, 16.5, '16-16.5'), (16.5, 99, '16.5+')]:
    mk = np.where((y >= lo) & (y < hi))[0]
    if len(mk):
        print(f'  定数[{tag}]: n={len(mk)} MAE={mean_absolute_error(y[mk], oof[mk]):.4f} bias={errs[mk].mean():+.4f}')

p95_vals, p99_vals = {}, {}
for j, name in enumerate(gb_feature_names):
    col = X_base[:, j]
    p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
    p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
_apply_jline_p95_fix(p95_vals)  # v12.1: 与app.py推理P95一致
boosts = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in feats_list])
X_all = np.hstack([X_base, X_lv])
scaler = StandardScaler().fit(X_all)
gb_final = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
gb_final.fit(scaler.transform(X_all), y - boosts, sample_weight=SAMPLE_W)
model = {
    'gb': gb_final, 'scaler': scaler, 'feature_names': gb_feature_names,
    'p95_vals': p95_vals, 'p99_vals': p99_vals, 'lv_order': LV_ORDER,
    'version': 'v12',
    'n_train': n, 'MANUAL_FLAT': FLAT, 'caps': CAPS,
    'train_meta': {'n': n, 'songs': len(set(names_list)),
                   'cv_mae': float(mean_absolute_error(y, oof)),
                   'cv_r2': float(r2_score(y, oof)),
                   'atw': args.atw, 'loww': args.loww,
                   'base': 'v11.13(train_v11_a.py atw1.0/loww1.5/none)',
                   'feat_fixes': 'v11.15e审计修复(单位/阈值/分音/jline位移量)',
                   'p95_fix': 'jline P95与app.py同步(107.1/18.6/15.1)'},
}
path = os.path.join(_ROOT, 'models', args.out)
with open(path, 'wb') as f:
    pickle.dump(model, f)
print(f'\n已保存: {path}')
