# -*- coding: utf-8 -*-
"""实验6b: 未上架5894谱 趋势分析 (清洗版: 社区定数 5<d<30)
"""
import os, sys, csv, numpy as np, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from scipy.stats import spearmanr

rows = []
with open(os.path.join(_ROOT, 'data', 'phira', 'unranked_4star_list.csv'), encoding='utf-8-sig') as f:
    rdr = csv.DictReader(f)
    for r_ in rdr:
        try:
            d = float(r_['difficulty']) if r_['difficulty'] else None
            if d is None or not (5.0 < d < 30.0): continue  # 清洗
            rows.append({
                'name': r_['name'], 'level': r_['level'], 'diff': d,
                'rating': float(r_['rating']), 'rc': int(r_['ratingCount']),
                'pred': float(r_['pred']), 'mf3': float(r_['mf3']),
            })
        except Exception:
            pass
print(f'未上架(清洗后): {len(rows)}')
ds = np.array([r['diff'] for r in rows]); ps = np.array([r['pred'] for r in rows]); rt = np.array([r['rating'] for r in rows])

rho_all, _ = spearmanr(ds, ps)
print(f'Spearman(社区定数~pred)={rho_all:.3f} bias={(ps-ds).mean():+.3f} MAE={np.abs(ps-ds).mean():.3f}')

print('\n分段bias (按社区定数):')
for lo, hi, tag in [(11,12,'11-12'),(12,13,'12-13'),(13,14,'13-14'),(14,15,'14-15'),(15,16,'15-16'),(16,17,'16-17'),(17,18,'17-18'),(18,99,'18+')]:
    mk = np.where((ds >= lo) & (ds < hi))[0]
    if len(mk) >= 10:
        print(f'  [{tag}]: n={len(mk)} bias={(ps[mk]-ds[mk]).mean():+.3f} MAE={np.abs(ps[mk]-ds[mk]).mean():.3f}')

# 多指/双指 分段
mf = np.array([r['mf3'] for r in rows])
print('\n多指/双指 段间与分段:')
for lbl, cond in [('多指(mf3>=30)', mf >= 30), ('双指(mf3<=5)', mf <= 5), ('混合', (mf > 5) & (mf < 30))]:
    g = np.where(cond)[0]
    if len(g) >= 30:
        rho, _ = spearmanr(ds[g], ps[g])
        print(f'  {lbl}: n={len(g)} Spearman={rho:.3f} bias={(ps[g]-ds[g]).mean():+.3f}')
        for lo, hi, tag in [(14,16,'14-16'),(16,99,'16+')]:
            mk = g[(ds[g] >= lo) & (ds[g] < hi)]
            if len(mk) >= 10:
                print(f'      [{tag}]: n={len(mk)} bias={(ps[mk]-ds[mk]).mean():+.3f}')

# 与上架谱对比: 同段预测分布
print('\n预测分布: 上架 vs 未上架 (同预测值段比例)')
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
import json
from boost_config import MANUAL_FLAT
m4 = pickle.load(open(os.path.join(_ROOT, 'models', '6dim_model_v11_4.pkl'), 'rb'))
gb, scaler = m4['gb'], m4['scaler']; FN = m4['feature_names']; LV_ORDER = m4['lv_order']
FLAT = m4['MANUAL_FLAT']; CAPS = m4['caps']; P95 = m4['p95_vals']; P99 = m4['p99_vals']
_ALIGN = json.load(open(os.path.join(_ROOT, 'data', 'domain_align.json'), encoding='utf-8')).get('delta', {})
MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
DENS_FEATS = {'above_avg_density_mean', 'real_core_notes_per_second'}
def predict(feats_raw, level='IN'):
    feats = dict(feats_raw); lv = level.upper()
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
    mf3 = feats_raw.get('multi_finger_3plus_events', 0); dens = feats_raw.get('above_avg_density_mean', 0)
    ml = feats_raw.get('multi_line_sim_events', 0); wmf = feats_raw.get('weighted_mf_score_per_sec', 0)
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
    return pred

ranked = [r for r in cache['ranked'] if r['diff'] and 5 < r['diff'] < 30]
prs = np.array([predict(r['feats'], r['level']) for r in ranked])
print(f'\n上架谱 n={len(ranked)}: 预测段分布')
for lo, hi, tag in [(10,14,'10-14'),(14,15,'14-15'),(15,16,'15-16'),(16,17,'16-17'),(17,18,'17-18'),(18,99,'18+')]:
    mk = np.where((prs >= lo) & (prs < hi))[0]
    print(f'  上架[{tag}]: {len(mk)} ({100*len(mk)/len(ranked):.1f}%)')
print('未上架(清洗后):')
for lo, hi, tag in [(10,14,'10-14'),(14,15,'14-15'),(15,16,'15-16'),(16,17,'16-17'),(17,18,'17-18'),(18,99,'18+')]:
    mk = np.where((ps >= lo) & (ps < hi))[0]
    print(f'  未上架[{tag}]: {len(mk)} ({100*len(mk)/len(ps):.1f}%)')
print('DONE')
