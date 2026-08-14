# -*- coding: utf-8 -*-
"""5折交叉验证: 刻画误差来源 (水平/档位/特征相关性) + 测试3个改进变体

变体:
  V0 基线:  GB(236特征) 残差=y-boost_v9;  预测=GB+boost_v9
  V1 加level: V0 + 4个level one-hot特征
  V2 无boost: GB直接预测y
  V3 无boost+level: GB直接预测y + level one-hot
"""
import os, sys, pickle, numpy as np, time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from app import MANUAL_FLAT

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
]
GB_KEEP = {'density_dimension', 'real_core_notes_per_second',
           'core_peak_density_1sec_top5avg', 'core_peak_density_top5avg_1beat'}
gb_feature_names = [nn for nn in feature_names
                    if nn in GB_KEEP or not any(kw in nn for kw in GB_EXCLUDE_KEYWORDS)]
print(f'GB特征数: {len(gb_feature_names)}')

X_base = np.array([[f.get(nn, 0) for nn in gb_feature_names] for f in feats_list])
y = np.array(labels)

# level one-hot: EZ HD IN AT
LV_ORDER = ['EZ', 'HD', 'IN', 'AT']
X_lv = np.zeros((n, 4))
for i, lv in enumerate(levels_list):
    X_lv[i, LV_ORDER.index(lv)] = 1.0

def compute_boost_v9(feats, p95_vals, p99_vals):
    total = 0.0
    for fname, bl, co in MANUAL_FLAT:
        v = feats.get(fname, 0)
        pv = p95_vals.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t: continue
        e = v / t - 1.0
        x = co * (e ** 0.70)
        if v > max(p99_vals.get(fname, 0), bl * 0.5):
            pe = v / max(p99_vals.get(fname, 0), bl * 0.5) - 1.0
            x += co * max(0, pe) ** 0.70 * 0.5
        total += x
    return total

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
splits = list(kf.split(X_base))

# out-of-fold predictions per variant
oof = {k: np.zeros(n) for k in ['V0', 'V1', 'V2', 'V3']}
t0 = time.time()

for fi, (tr, te) in enumerate(splits):
    t_fold = time.time()
    # P95/P99 从训练折
    p95_vals, p99_vals = {}, {}
    for j, name in enumerate(gb_feature_names):
        col = X_base[tr, j]
        p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
        p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0

    boosts = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in feats_list])
    b_tr, b_te = boosts[tr], boosts[te]

    X_tr = X_base[tr]; X_te = X_base[te]
    X_tr_lv = np.hstack([X_tr, X_lv[tr]]); X_te_lv = np.hstack([X_te, X_lv[te]])

    scalers = {}
    for key, Xtr in [('s0', X_tr), ('s1', X_tr_lv)]:
        sc = StandardScaler(); scalers[key] = sc.fit(Xtr)

    y_res_tr = y[tr] - b_tr

    # V0: GB残差 + boost
    g0 = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
    g0.fit(scalers['s0'].transform(X_tr), y_res_tr)
    oof['V0'][te] = g0.predict(scalers['s0'].transform(X_te)) + b_te

    # V1: GB残差+level + boost
    g1 = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
    g1.fit(scalers['s1'].transform(X_tr_lv), y_res_tr)
    oof['V1'][te] = g1.predict(scalers['s1'].transform(X_te_lv)) + b_te

    # V2: GB直接预测y (无boost)
    g2 = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
    g2.fit(scalers['s0'].transform(X_tr), y[tr])
    oof['V2'][te] = g2.predict(scalers['s0'].transform(X_te))

    # V3: GB直接预测y + level
    g3 = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
    g3.fit(scalers['s1'].transform(X_tr_lv), y[tr])
    oof['V3'][te] = g3.predict(scalers['s1'].transform(X_te_lv))

    print(f'  fold{fi}: 完成 ({time.time()-t_fold:.0f}s)', flush=True)

print(f'\n总耗时: {(time.time()-t0)/60:.1f} 分钟')

# ====== 汇总 ======
print('\n' + '='*70)
print('5折CV 样本外结果 (n=%d)' % n)
print('='*70)
for k in ['V0', 'V1', 'V2', 'V3']:
    err = oof[k] - y
    print(f'{k}: R2={r2_score(y, oof[k]):.4f}  MAE={mean_absolute_error(y, oof[k]):.4f}  '
          f'偏差={np.mean(err):+.3f}  RMSE={np.sqrt(np.mean(err**2)):.4f}')

# 按level分析V0
print('\n=== V0 按level ===')
for lv in LV_ORDER:
    m = np.array([i for i in range(n) if levels_list[i] == lv])
    err = oof['V0'][m] - y[m]
    print(f'  {lv}: n={len(m):<4} MAE={mean_absolute_error(y[m], oof["V0"][m]):.4f} '
          f'偏差={np.mean(err):+.3f} 中位={np.median(err):+.3f}')

# 按真定数档位
print('\n=== V0 按真定数档位 ===')
for name, lo, hi in [('<4',0,4),('4-7',4,7),('7-11',7,11),('11-14',11,14),('14-16.5',14,16.5),('>16.5',16.5,99)]:
    m = np.where((y >= lo) & (y < hi))[0]
    if len(m) == 0: continue
    err = oof['V0'][m] - y[m]
    print(f'  [{name}]: n={len(m):<3} MAE={mean_absolute_error(y[m], oof["V0"][m]):.3f} 偏差={np.mean(err):+.3f}')

# 每个变体按level
print('\n=== 各变体按level MAE/偏差 ===')
for k in ['V0','V1','V2','V3']:
    line = f'  {k}: '
    for lv in LV_ORDER:
        m = np.array([i for i in range(n) if levels_list[i] == lv])
        err = oof[k][m] - y[m]
        line += f'{lv}:{mean_absolute_error(y[m], oof[k][m]):.2f}({np.mean(err):+.2f})  '
    print(line)

# V0 最大误差谱面
print('\n=== V0 高估Top8 ===')
for i in np.argsort(oof['V0'] - y)[::-1][:8]:
    print(f'  {names_list[i]:<32} {levels_list[i]:<3} 真={y[i]:.1f} 预测={oof["V0"][i]:.2f} 偏差={oof["V0"][i]-y[i]:+.2f}')
print('=== V0 低估Top8 ===')
for i in np.argsort(oof['V0'] - y)[:8]:
    print(f'  {names_list[i]:<32} {levels_list[i]:<3} 真={y[i]:.1f} 预测={oof["V0"][i]:.2f} 偏差={oof["V0"][i]-y[i]:+.2f}')

# 保存结果供后续分析
np.savez(os.path.join(_ROOT, 'tools', 'cv_oof.npz'),
         oof_v0=oof['V0'], oof_v1=oof['V1'], oof_v2=oof['V2'], oof_v3=oof['V3'],
         y=y, names=np.array(names_list), levels=np.array(levels_list))
print('\n结果已保存 tools/cv_oof.npz')
