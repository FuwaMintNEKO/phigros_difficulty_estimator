# -*- coding: utf-8 -*-
"""v10 训练/评估: 979官谱 (含6首新歌) + V1设计 (GB残差+boost+level特征)
   + 按level isotonic校准 (诚实: 校准器只在本折以外拟合)

评估指标: 总体/按level/按定数档位的 MAE, R2, 偏差
同时测试 robust 特征变换 (speed_volatility 等 log1p) 的影响
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_absolute_error
from boost_config import MANUAL_FLAT

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

# level one-hot
LV_ORDER = ['EZ', 'HD', 'IN', 'AT']
X_lv = np.zeros((n, 4))
for i, lv in enumerate(levels_list):
    X_lv[i, LV_ORDER.index(lv)] = 1.0

# robust 变换: speed_volatility 等极端值特征做 log1p
ROBUST_FEATS = ['speed_volatility', 'speed_std', 'speed_event_density',
                'jline_movement_density', 'jline_rotate_density', 'jline_disappear_density']
robust_idx = [gb_feature_names.index(r) for r in ROBUST_FEATS if r in gb_feature_names]
X_robust = X_base.copy()
for j in robust_idx:
    X_robust[:, j] = np.log1p(X_robust[:, j])

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

def run_variant(Xmat, label, do_calib=True):
    """对给定特征矩阵跑 5折: GB残差+level; 返回OOF及校准后的OOF"""
    oof = np.zeros(n)
    oof_cal = np.zeros(n)
    t0 = time.time()
    for fi, (tr, te) in enumerate(splits):
        p95_vals, p99_vals = {}, {}
        for j, name in enumerate(gb_feature_names):
            col = Xmat[tr, j]
            p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
            p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
        boosts = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in feats_list])
        b_tr, b_te = boosts[tr], boosts[te]

        X_tr = np.hstack([Xmat[tr], X_lv[tr]])
        X_te = np.hstack([Xmat[te], X_lv[te]])
        sc = StandardScaler().fit(X_tr)
        gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                       learning_rate=0.05, subsample=0.8, random_state=42)
        gb.fit(sc.transform(X_tr), y[tr] - b_tr)
        oof[te] = gb.predict(sc.transform(X_te)) + b_te
        print(f'  [{label}] fold{fi} 完成 ({time.time()-t0:.0f}s)', flush=True)

    if do_calib:
        # 诚实校准: 对每个fold, 用其它fold的OOF预测拟合按level的isotonic, 应用到本fold
        levels_arr = np.array(levels_list)
        for fi, (tr, te) in enumerate(splits):
            for lv in LV_ORDER:
                m_tr = np.where((levels_arr == lv) & (np.isin(np.arange(n), tr)))[0]
                m_te = np.where((levels_arr == lv) & (np.isin(np.arange(n), te)))[0]
                if len(m_tr) < 10 or len(m_te) == 0:
                    oof_cal[m_te] = oof[m_te]
                    continue
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(oof[m_tr], y[m_tr])
                oof_cal[m_te] = iso.predict(oof[m_te])
        return oof, oof_cal
    return oof, oof

# ===== 跑变体 =====
print('>>> V10a: 原始特征 + level + 按level校准')
oof_a, oof_a_cal = run_variant(X_base, 'v10a')
print('>>> V10b: robust特征(log1p) + level + 按level校准')
oof_b, oof_b_cal = run_variant(X_robust, 'v10b')

# ===== 汇总 =====
np.savez(os.path.join(_ROOT, 'tools', 'cv_oof_v10.npz'),
         oof_a=oof_a, oof_a_cal=oof_a_cal, oof_b=oof_b, oof_b_cal=oof_b_cal,
         y=y, names=np.array(names_list), levels=np.array(levels_list))

print('\n' + '='*78)
print(f'5折CV 样本外 (n={n}, 含6首新歌)')
print('='*78)
for label, pred in [('V10a 原始特征(未校准)', oof_a),
                    ('V10a 按level校准', oof_a_cal),
                    ('V10b robust+校准', oof_b_cal)]:
    err = pred - y
    print(f'{label:<22} R2={r2_score(y, pred):.4f}  MAE={mean_absolute_error(y, pred):.4f}  '
          f'偏差={np.mean(err):+.3f}  RMSE={np.sqrt(np.mean(err**2)):.4f}')

print('\n=== 按level (V10a 校准后) ===')
for lv in LV_ORDER:
    m = np.where(np.array(levels_list) == lv)[0]
    err = oof_a_cal[m] - y[m]
    print(f'  {lv}: n={len(m):<4} MAE={mean_absolute_error(y[m], oof_a_cal[m]):.4f} 偏差={np.mean(err):+.3f}')

print('\n=== 按定数档位 (V10a 校准后) ===')
for name, lo, hi in [('<4',0,4),('4-7',4,7),('7-11',7,11),('11-14',11,14),('14-16.5',14,16.5),('>16.5',16.5,99)]:
    m = np.where((y >= lo) & (y < hi))[0]
    if len(m) == 0: continue
    err = oof_a_cal[m] - y[m]
    print(f'  [{name}]: n={len(m):<3} MAE={mean_absolute_error(y[m], oof_a_cal[m]):.3f} 偏差={np.mean(err):+.3f}')

print('\n=== V10a校准后 高估Top6 / 低估Top6 ===')
for i in np.argsort(oof_a_cal - y)[::-1][:6]:
    print(f'  +{oof_a_cal[i]-y[i]:+.2f}  {names_list[i]:<34} {levels_list[i]:<3} 真={y[i]:.1f} 预测={oof_a_cal[i]:.2f}')
for i in np.argsort(oof_a_cal - y)[:6]:
    print(f'  {oof_a_cal[i]-y[i]:+.2f}  {names_list[i]:<34} {levels_list[i]:<3} 真={y[i]:.1f} 预测={oof_a_cal[i]:.2f}')
