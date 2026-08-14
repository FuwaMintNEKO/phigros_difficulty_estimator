# -*- coding: utf-8 -*-
"""上架谱偏差 vs 特征偏移的相关性: 找出高估主因特征"""
import os, sys, pickle, numpy as np, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT

with open(os.path.join(_ROOT, 'models', '6dim_model_v11_1.pkl'), 'rb') as f:
    m = pickle.load(f)
gb, scaler = m['gb'], m['scaler']
FN, P95, P99 = m['feature_names'], m['p95_vals'], m['p99_vals']
LV_ORDER = m.get('lv_order', ['EZ','HD','IN','AT'])
CAPS = m.get('caps', {})
FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT)
_ALIGN = json.load(open(os.path.join(_ROOT, 'data', 'domain_align.json'), encoding='utf-8')).get('delta', {})

MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}

def predict(feats_raw, level='IN'):
    feats = dict(feats_raw)
    lv = level.upper()
    if 'IN_AT' in LV_ORDER and lv in ('IN','AT'): lv = 'IN_AT'
    if lv not in LV_ORDER: lv = LV_ORDER[-1]
    vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(lv)] = 1.0
    x = np.array([[feats.get(n,0) for n in FN] + vec])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    mf3 = feats_raw.get('multi_finger_3plus_events', 0)
    dens = feats_raw.get('above_avg_density_mean', 0)
    mf_scale = (0.70 if dens >= 12.5 else 0.50) if mf3 >= 30 else (1.0 if mf3 <= 5 else 0.8)
    eff_scale = 1.0 if mf3 >= 30 else (1.5 if mf3 <= 5 else 1.0)
    total = 0.0
    cd_ = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0)
        pv = P95.get(fname, 0)
        t = max(pv*0.55, bl*0.5)
        if v <= t: continue
        e = v/t - 1.0
        c = CAPS.get(fname, cd_)
        if c is not None and e > c: e = c
        co2 = co * (mf_scale if fname in MF_FEATS else (eff_scale if fname in EFF_FEATS else 1.0))
        x_ = co2 * (e**0.70)
        p99 = max(P99.get(fname,0), bl*0.5)
        if v > p99:
            pe = v/p99 - 1.0
            if c is not None and pe > c: pe = c
            x_ += co2*max(0,pe)**0.70*0.5
        total += x_
    return p_gb + total

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
valid = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]

# 预测
for r in valid:
    r['pred'] = predict(r['feats'], r['level'])
    r['err'] = r['pred'] - r['diff']

# 相关性: err vs 特征
feat_list = ['type_switch_per_sec', 'judge_line_count', 'offbeat_ratio', 'hold_tap_overlap_count',
             'above_avg_density_mean', 'real_core_notes_per_second', 'eff_avg_tps_1s', 'eff_peak_tps_1s',
             'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'multi_line_sim_events',
             'stair_speed_avg', 'movement_per_second', 'speed_volatility', 'tempo_change_count',
             'total_notes', 'duration_sec', 'rhythm_entropy', 'fast_note_density_32nd', 'chord_alternation_rate']
print('=== 偏差(err) 与特征的相关性 (14-17段, 控制diff后) ===')
seg = [r for r in valid if 14 <= r['diff'] <= 17]
errs = np.array([r['err'] for r in seg])
diffs = np.array([r['diff'] for r in seg])
print(f'n={len(seg)}')
for k in feat_list:
    vals = np.array([r['feats'].get(k, 0) for r in seg])
    if vals.std() < 1e-9: continue
    # 残差相关 (去掉diff影响)
    r1 = np.corrcoef(vals, errs)[0,1]
    # 偏相关: 控制diff
    A = np.column_stack([diffs, np.ones(len(diffs))])
    resid_err = errs - A @ np.linalg.lstsq(A, errs, rcond=None)[0]
    resid_val = vals - A @ np.linalg.lstsq(A, vals, rcond=None)[0]
    rp = np.corrcoef(resid_val, resid_err)[0,1]
    print(f'  {k:<30} r(err)={r1:+.3f} 偏r(控diff)={rp:+.3f}')

# 高估谱的特征模式: 偏差>0.5 的谱 vs 全体的特征对比
over = [r for r in seg if r['err'] > 0.5]
under = [r for r in seg if r['err'] < -0.5]
print(f'\n高估(>0.5): {len(over)} | 低估(<-0.5): {len(under)}')
print('\n=== 特征均值对比: 高估 vs 全体 vs 低估 ===')
for k in feat_list:
    vo = np.mean([r['feats'].get(k,0) for r in over]) if over else 0
    va = np.mean([r['feats'].get(k,0) for r in seg])
    vu = np.mean([r['feats'].get(k,0) for r in under]) if under else 0
    if abs(vo-va) > 0.05*max(va, 0.01) or abs(vu-va) > 0.05*max(va, 0.01):
        print(f'  {k:<30} 高估={vo:>10.2f} 全体={va:>10.2f} 低估={vu:>10.2f}')
