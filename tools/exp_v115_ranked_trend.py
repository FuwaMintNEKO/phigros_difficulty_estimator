# -*- coding: utf-8 -*-
"""实验4: 上架589谱 段内Spearman趋势分析 (全量模型对社区谱=严格外推)
"""
import os, sys, pickle, numpy as np, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from scipy.stats import spearmanr
from boost_config import MANUAL_FLAT

m4 = pickle.load(open(os.path.join(_ROOT, 'models', '6dim_model_v11_4.pkl'), 'rb'))
gb, scaler = m4['gb'], m4['scaler']
FN = m4['feature_names']; LV_ORDER = m4['lv_order']
FLAT = m4['MANUAL_FLAT']; CAPS = m4['caps']; P95 = m4['p95_vals']; P99 = m4['p99_vals']
_ALIGN = json.load(open(os.path.join(_ROOT, 'data', 'domain_align.json'), encoding='utf-8')).get('delta', {})
MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
DENS_FEATS = {'above_avg_density_mean', 'real_core_notes_per_second'}

def predict(feats_raw, level='IN', calib=0):
    feats = dict(feats_raw)
    lv = level.upper()
    if 'AT' in lv: lv = 'AT'
    elif 'IN' in lv: lv = 'IN'
    elif 'HD' in lv: lv = 'HD'
    else: lv = 'IN'
    if lv == 'IN':
        for k, d in _ALIGN.items():
            if k in feats: feats[k] = feats[k] - d
    if 'IN_AT' in LV_ORDER and lv in ('IN','AT'): lv = 'IN_AT'
    if lv not in LV_ORDER: lv = LV_ORDER[-1]
    vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(lv)] = 1.0
    x = np.array([[feats.get(n,0) for n in FN] + vec])
    p_gb = float(gb.predict(scaler.transform(x))[0])
    mf3 = feats_raw.get('multi_finger_3plus_events', 0)
    dens = feats_raw.get('above_avg_density_mean', 0)
    ml = feats_raw.get('multi_line_sim_events', 0)
    wmf = feats_raw.get('weighted_mf_score_per_sec', 0)
    if mf3 >= 30 and ml >= 100: mf_scale, dens_s = 0.45, 0.85
    elif mf3 >= 30: mf_scale, dens_s = (0.70 if dens >= 9.5 else 0.50), 1.0
    else: mf_scale, dens_s = (1.0 if mf3 <= 5 else 0.8), 1.0
    if mf3 <= 5:
        _sw = min(max((wmf - 12.0) / 6.0, 0.0), 1.0)
        eff_scale = 1.0 if dens >= 10.0 else 1.5 - 0.5 * _sw
        wmf_scale = 1.0 - 0.4 * _sw
    else: eff_scale, wmf_scale = 1.0, 1.0
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
        r4 = feats_raw.get('tracks_4plus_sec', 0)/act; r5 = feats_raw.get('tracks_5plus_sec', 0)/act; r6 = feats_raw.get('tracks_6plus_sec', 0)/act
        pred += 0.15*min(r4,0.8) + 0.55*min(r5,0.4) + 1.0*min(r6,0.15)
    for lo, hi, adj in [(14,15,0.40),(15,16,0.25),(16,17,0.05)]:
        if lo < pred <= hi: pred -= adj; break
    return pred, p_gb, total

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
print(f'上架谱: {len(ranked)}')

ds = np.array([r['diff'] for r in ranked])
ps = np.array([predict(r['feats'], r['level'])[0] for r in ranked])
errs = ps - ds
rho_all, _ = spearmanr(ds, ps)
print(f'整体: n={len(ranked)} Spearman={rho_all:.3f} bias={errs.mean():+.3f} MAE={np.abs(errs).mean():.3f}')

print('\n===== 段内 Spearman (社区谱=外推) =====')
for lo, hi, tag in [(11,14,'11-14'),(14,15,'14-15'),(15,16,'15-16'),(16,17,'16-17'),(17,99,'17+')]:
    mk = np.where((ds >= lo) & (ds < hi))[0]
    if len(mk) < 6: continue
    rho, _ = spearmanr(ds[mk], ps[mk])
    print(f'  [{tag}]: n={len(mk)} Spearman={rho:.3f} bias={errs[mk].mean():+.3f}')

print('\n===== 按难度段 多指/双指 段内 Spearman =====')
for lo, hi, tag in [(14,16,'14-16'),(16,99,'16+')]:
    mk = np.where((ds >= lo) & (ds < hi))[0]
    for lbl, cond in [('多指', lambda i: ranked[i]['feats'].get('multi_finger_3plus_events',0)>=30),
                      ('双指', lambda i: ranked[i]['feats'].get('multi_finger_3plus_events',0)<=5),
                      ('混合', lambda i: 5 < ranked[i]['feats'].get('multi_finger_3plus_events',0) < 30)]:
        g = [i for i in mk if cond(i)]
        if len(g) >= 6:
            rho, _ = spearmanr(ds[g], ps[g])
            print(f'  [{tag}] {lbl}: n={len(g)} Spearman={rho:.3f} bias={(ps[g]-ds[g]).mean():+.3f}')

# GB 部分 vs Boost 部分 的趋势贡献
print('\n===== GB-only vs Full 的 Spearman (上架谱) =====')
gb_only = np.array([predict(r['feats'], r['level'])[1] for r in ranked])
rho_gb, _ = spearmanr(ds, gb_only)
print(f'  GB-only: Spearman={rho_gb:.3f}')
print('DONE')
