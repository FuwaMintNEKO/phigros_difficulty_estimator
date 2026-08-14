# -*- coding: utf-8 -*-
"""16.5+ 段逐谱诊断: 无降权最终预测 vs 真实"""
import os, sys, numpy as np, io, pickle
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

def predict_full(feats_raw, level, stack_scale_fn=None, calib=[(14,15,0.41),(15,16,0.26),(16,17,0.06)]):
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
    ts = app_mod.compute_tags(feats)
    high = {'叠键', '多押', '变速', '位移'}
    is_stack = sum(1 for t in ts if t in high) >= 2
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
        if stack_scale_fn: co2 = co2 * stack_scale_fn(fname, v)
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
    return pred, p_gb, total, is_stack

rows = []
for i, r in enumerate(ranked):
    if ds[i] >= 16.5:
        pr, pgb, tot, st = predict_full(r['feats'], r['level'])
        rows.append((r['name'][:26], ds[i], pr, pgb, tot, st, round(pr-ds[i],2)))
rows.sort(key=lambda x: x[6])
print(f'{"谱名":<28}{"真实":>6}{"预测":>6}{"gb":>7}{"boost":>7}{"堆":>3}{"err":>7}')
for nm, d, pr, pgb, tot, st, e in rows:
    print(f'{nm:<28}{d:>6.2f}{pr:>6.2f}{pgb:>7.2f}{tot:>7.2f}{"Y" if st else "-":>3}{e:>+7.2f}')
errs = np.array([x[6] for x in rows])
print(f'\n段统计: n={len(rows)} bias={errs.mean():+.3f} MAE={np.abs(errs).mean():.3f}')
print('DONE')