# -*- coding: utf-8 -*-
"""细段校准表扫描: 14-15/15-16/16-16.5/16.5-17/17+ 五段独立优化 (无堆料降权)"""
import os, sys, numpy as np, io, pickle, itertools
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
gb, scaler = app_mod.gb, app_mod.scaler
FN, LV_ORDER = app_mod.FN, app_mod.LV_ORDER
_ALIGN = app_mod.DOMAIN_DELTA
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])

def predict_base(feats_raw, level, calib):
    """无降权完整预测"""
    feats = dict(feats_raw)
    lv = level.upper()
    if 'AT' in lv: lv = 'AT'
    elif 'IN' in lv: lv = 'IN'
    elif 'HD' in lv: lv = 'HD'
    else: lv = 'IN'
    if lv == 'IN':
        for k, d in _ALIGN.items():
            if k in feats: feats[k] = feats[k] - d
    lv2 = 'IN_AT' if lv in ('IN','AT') and 'IN_AT' in LV_ORDER else lv
    if lv2 not in LV_ORDER: lv2 = LV_ORDER[-1]
    vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(lv2)] = 1.0
    x = np.array([[feats.get(n,0) for n in FN] + vec])
    p_gb = float(gb.predict(scaler.transform(x))[0])
    FLAT = app_mod.MANUAL_FLAT; P95 = app_mod.P95; P99 = app_mod.P99; CAPS = app_mod.CAPS
    MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
    EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
    DENS_FEATS = {'above_avg_density_mean', 'real_core_notes_per_second'}
    EXTREME_FEATS = {'cross_hand_density', 'jline_relative_cross', 'thirtysecond_run_max', 'thirtysecond_run_ratio', 'lane_switch_density'}
    mf3 = feats_raw.get('multi_finger_3plus_events', 0); dens = feats_raw.get('above_avg_density_mean', 0)
    ml = feats_raw.get('multi_line_sim_events', 0); wmf = feats_raw.get('weighted_mf_score_per_sec', 0)
    if mf3 >= 30 and ml >= 100: mf_scale, dens_s = 0.45, 0.85
    elif mf3 >= 30: mf_scale, dens_s = (0.70 if dens >= 9.5 else 0.50), 1.0
    else: mf_scale, dens_s = (1.0 if mf3 <= 5 else 0.8), 1.0
    if mf3 <= 5:
        _sw = min(max((wmf - 12.0) / 6.0, 0.0), 1.0)
        eff_scale = 1.0 if dens >= 10.0 else 1.5 - 0.5 * _sw
        wmf_scale = 1.0 - 0.4 * _sw
        extreme_scale = 1.3
    elif mf3 >= 30:
        eff_scale, wmf_scale = 1.0, 1.0
        extreme_scale = 0.7
    else:
        eff_scale, wmf_scale = 1.0, 1.0
        extreme_scale = 1.0
    total = 0.0; cd_ = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0); pv = P95.get(fname, 0)
        t = max(pv*0.55, bl*0.5)
        if v <= t: continue
        e = v/t - 1.0
        c = CAPS.get(fname, cd_)
        if c is not None and e > c: e = c
        co2 = co
        if fname in MF_FEATS: co2 = co * mf_scale
        elif fname in EFF_FEATS: co2 = co * eff_scale
        if fname in DENS_FEATS and mf3 >= 30 and ml >= 100: co2 = co * dens_s
        if fname == 'weighted_mf_score_per_sec': co2 = co * wmf_scale
        if fname in EXTREME_FEATS: co2 = co * extreme_scale
        x_ = co2 * (e**0.70)
        p99 = max(P99.get(fname,0), bl*0.5)
        if v > p99:
            pe = v/p99 - 1.0
            if c is not None and pe > c: pe = c
            x_ += co2*max(0,pe)**0.70*0.5
        total += x_
    pred = p_gb + total
    act = feats_raw.get('tracks_active_sec', 0)
    if act > 0:
        pred += 0.15*min(feats_raw.get('tracks_4plus_sec',0)/act,0.8) + 0.55*min(feats_raw.get('tracks_5plus_sec',0)/act,0.4) + 1.0*min(feats_raw.get('tracks_6plus_sec',0)/act,0.15)
    hr = feats_raw.get('hold_count', 0) / max(feats_raw.get('total_notes', 1), 1)
    if hr >= 0.6: pred += 0.7
    elif hr >= 0.4: pred += 0.5
    elif hr >= 0.25: pred += 0.3
    for lo, hi, adj in calib:
        if lo < pred <= hi: pred -= adj; break
    return pred

# 先算无校准预测, 再对校准做网格 (加速: 预计算)
preds_raw = np.array([predict_base(r['feats'], r['level'], []) for r in ranked])
def apply_calib(ps, calib):
    out = ps.copy()
    for lo, hi, adj in calib:
        m = (ps > lo) & (ps <= hi)
        out[m] -= adj
    return out

# 网格: a=14-15, b=15-16, c=16-16.5, d=16.5-17, e=17+
best = None
for a, b, c, d, e in itertools.product([0.36,0.41,0.46], [0.21,0.26,0.31], [0.16,0.26,0.36,0.46], [0.0,0.06,0.16,0.26], [0.0,0.06]):
    calib = [(14,15,a),(15,16,b),(16,16.5,c),(16.5,17,d),(17,99,e)]
    ps = apply_calib(preds_raw, calib)
    errs = ps - ds
    segs = []
    for lo, hi, t2 in [(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
        mk = np.where((ds >= lo) & (ds < hi))[0]
        if len(mk): segs.append((t2, errs[mk].mean()))
    mae = np.abs(errs).mean()
    # 目标: 各段bias尽量小
    score = mae + 0.8*np.std([s[1] for s in segs]) + 0.5*abs(errs[ds>=16.5].mean())
    if best is None or score < best[0]:
        best = (score, mae, (a,b,c,d,e), segs)
score, mae, (a,b,c,d,e), segs = best
print(f'best: a={a} b={b} c={c} d={d} e={e} MAE={mae:.3f}')
print('segs:', ' '.join(f'{t}:{v:+.3f}' for t,v in segs))
# 打印几个次优
res = []
for a, b, c, d, e in itertools.product([0.41], [0.26], [0.16,0.26,0.36,0.46], [0.0,0.06,0.16,0.26], [0.0,0.06]):
    calib = [(14,15,a),(15,16,b),(16,16.5,c),(16.5,17,d),(17,99,e)]
    ps = apply_calib(preds_raw, calib)
    errs = ps - ds
    segs = []
    for lo, hi, t2 in [(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
        mk = np.where((ds >= lo) & (ds < hi))[0]
        if len(mk): segs.append(f'{t2}:{errs[mk].mean():+.3f}')
    res.append((np.abs(errs).mean(), c, d, e, ' '.join(segs)))
res.sort()
print('\n(固定a=0.41 b=0.26) top8:')
for mae, c, d, e, s in res[:8]:
    print(f'  c={c} d={d} e={e} MAE={mae:.3f} | {s}')
print('DONE')