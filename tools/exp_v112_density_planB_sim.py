# -*- coding: utf-8 -*-
"""t2 方案B全量模拟: eff版 above_avg_density_mean 替换后, v11.1 预测偏差变化
官谱982(已重算) + 上架589(json重算密度部分) 
对比: 原始above_avg vs eff版above_avg 的 pred-diff
"""
import os, sys, pickle, json
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
_ROOT = r'D:\\Trae项目\\新建文件夹\\phigros_difficulty_estimator'
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT
from unified_parser import load_chart_from_bytes
from tools.exp_v112_density_planB import eff_density_features

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
DENS_FEATS = {'above_avg_density_mean', 'real_core_notes_per_second'}

def predict(feats_raw, level='IN', dens_override=None):
    feats = dict(feats_raw)
    if level.upper() == 'IN':
        for k, d in _ALIGN.items():
            if k in feats: feats[k] = feats[k] - d
    if dens_override is not None:
        feats['above_avg_density_mean'] = dens_override
    lv = level.upper()
    if 'IN_AT' in LV_ORDER and lv in ('IN','AT'): lv = 'IN_AT'
    if lv not in LV_ORDER: lv = LV_ORDER[-1]
    vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(lv)] = 1.0
    x = np.array([[feats.get(n,0) for n in FN] + vec])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    mf3 = feats_raw.get('multi_finger_3plus_events', 0)
    mf4 = feats_raw.get('multi_finger_4plus_events', 0)
    ml = feats_raw.get('multi_line_sim_events', 0)
    dens = feats.get('above_avg_density_mean', 0)
    effa = feats_raw.get('eff_avg_tps_1s', 0)
    rcnps = feats_raw.get('real_core_notes_per_second', 0)
    eff_ratio = effa / max(dens, 0.1)
    if mf3 >= 30:
        if ml >= 100:
            mf_scale, dens_scale, dens_cap = 0.40, 0.80, None
        elif eff_ratio < 0.40 and mf4 >= 30:
            mf_scale, dens_scale, dens_cap = 0.55, 0.70, None
        else:
            mf_scale, dens_scale, dens_cap = 0.75, 1.0, (1.0 if rcnps >= 12 else None)
    else:
        mf_scale = 1.0 if mf3 <= 5 else 0.8
        dens_scale, dens_cap = 1.0, None
    eff_scale = 1.0 if mf3 >= 30 else (1.5 if mf3 <= 5 else 1.0)
    wmf_scale = 0.5 if mf3 <= 5 else 1.0
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
        co2 = co
        if fname in MF_FEATS: co2 = co * mf_scale
        elif fname in EFF_FEATS: co2 = co * eff_scale
        elif fname in DENS_FEATS:
            co2 = co * dens_scale
            if dens_cap is not None and fname == 'above_avg_density_mean':
                if e > dens_cap: e = dens_cap
        if fname == 'weighted_mf_score_per_sec': co2 = co * wmf_scale
        x_ = co2 * (e**0.70)
        p99 = max(P99.get(fname,0), bl*0.5)
        if v > p99:
            pe = v/p99 - 1.0
            if c is not None and pe > c: pe = c
            x_ += co2*max(0,pe)**0.70*0.5
        total += x_
    pred = p_gb + total
    if 14 < pred <= 15: pred -= 0.30
    elif 15 < pred <= 16: pred -= 0.18
    elif 16 < pred <= 17: pred -= 0.05
    return pred

# ===== 官谱: 用已保存的 eff 版重算结果 =====
with open(os.path.join(_ROOT, 'tools', '_tmp_planB_results.pkl'), 'rb') as f:
    planB_rows = pickle.load(f)
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)

results = []  # (name, src, diff, pred_orig, pred_eff)
pb_map = {}
for r in planB_rows:
    pb_map[(r['name'], r['level'])] = r
for o in cache['official']:
    key = (o['name'], o['level'])
    pb = pb_map.get(key)
    if not pb or not o.get('diff'): continue
    diff = float(o['diff'])
    po = predict(o['feats'], o['level'])
    pe = predict(o['feats'], o['level'], dens_override=pb['above_avg_eff'])
    results.append({'name': o['name'], 'level': o['level'], 'src': 'official',
                    'diff': diff, 'pred_orig': po, 'pred_eff': pe,
                    'dens_orig': pb['above_avg_orig'], 'dens_eff': pb['above_avg_eff']})

# ===== 上架谱: 重算 eff 版 =====
JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json')
ranked_meta = {}
charts = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
for lst in charts.values():
    for c in lst:
        ranked_meta[c['id']] = c
n_rank = 0
for r_ in cache['ranked']:
    if not r_.get('diff') or r_['diff'] <= 10: continue
    cid = r_['id']
    fn = os.path.join(JSON_DIR, f'{cid}.json')
    if not os.path.exists(fn): continue
    try:
        with open(fn, 'rb') as f:
            cd, raw = load_chart_from_bytes(f.read())
        pb = eff_density_features(cd)
        if not pb: continue
        diff = float(r_['diff'])
        po = predict(r_['feats'], r_['level'])
        pe = predict(r_['feats'], r_['level'], dens_override=pb['above_avg_eff'])
        results.append({'name': r_['name'], 'level': r_['level'], 'src': 'ranked',
                        'diff': diff, 'pred_orig': po, 'pred_eff': pe,
                        'dens_orig': pb['above_avg_orig'], 'dens_eff': pb['above_avg_eff']})
        n_rank += 1
    except Exception:
        pass
print(f'官谱 {sum(1 for r in results if r["src"]=="official")} 上架 {sum(1 for r in results if r["src"]=="ranked")}')

print()
print('===== 偏差统计: 原始 vs eff版 above_avg =====')
for src in ['official', 'ranked']:
    sel = [r for r in results if r['src'] == src]
    bo = np.mean([r['pred_orig']-r['diff'] for r in sel])
    be = np.mean([r['pred_eff']-r['diff'] for r in sel])
    mao = np.mean([abs(r['pred_orig']-r['diff']) for r in sel])
    mae = np.mean([abs(r['pred_eff']-r['diff']) for r in sel])
    print(f'  {src:<10} n={len(sel):<4} bias: {bo:+.3f} -> {be:+.3f} (Δ{be-bo:+.3f})   MAE: {mao:.3f} -> {mae:.3f}')

print()
print('===== 按定数段: 上架谱 =====')
bins = [('13-14', 13, 14), ('14-15', 14, 15), ('15-16', 15, 16), ('16-17', 16, 17), ('>=17', 17, 99)]
for name, lo, hi in bins:
    sel = [r for r in results if r['src']=='ranked' and lo <= r['diff'] < hi]
    if not sel: continue
    bo = np.mean([r['pred_orig']-r['diff'] for r in sel])
    be = np.mean([r['pred_eff']-r['diff'] for r in sel])
    print(f'  {name:<6} n={len(sel):<4} bias: {bo:+.3f} -> {be:+.3f} (Δ{be-bo:+.3f})')

print()
print('===== 按定数段: 官谱 (回归影响) =====')
for name, lo, hi in bins:
    sel = [r for r in results if r['src']=='official' and lo <= r['diff'] < hi]
    if not sel: continue
    bo = np.mean([r['pred_orig']-r['diff'] for r in sel])
    be = np.mean([r['pred_eff']-r['diff'] for r in sel])
    print(f'  {name:<6} n={len(sel):<4} bias: {bo:+.3f} -> {be:+.3f} (Δ{be-bo:+.3f})')

print()
print('===== eff版使预测下降最大的 15 张上架谱 (多押撑密度修正效果) =====')
for r in sorted(results, key=lambda x: x['pred_orig']-x['pred_eff'])[:15]:
    if r['src'] != 'ranked': continue
    print(f'  {r["name"][:24]:<26} diff={r["diff"]:>5.1f} pred {r["pred_orig"]:.2f}->{r["pred_eff"]:.2f} (Δ{r["pred_eff"]-r["pred_orig"]:+.2f}) dens {r["dens_orig"]:.1f}->{r["dens_eff"]:.1f}')

print()
print('===== 问题谱例 (若有) =====')
for kw in ['ギザ', 'Sigma']:
    for r in results:
        if kw.lower() in r['name'].lower():
            print(f'  [{r["src"]}] {r["name"][:40]} diff={r["diff"]} pred {r["pred_orig"]:.2f}->{r["pred_eff"]:.2f}')

with open(os.path.join(_ROOT, 'tools', '_tmp_planB_sim.pkl'), 'wb') as f:
    pickle.dump(results, f)
print()
print('已保存 tools/_tmp_planB_sim.pkl')
