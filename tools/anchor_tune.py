# -*- coding: utf-8 -*-
"""组合权重实验: 用5锚点找最优权重调整 (drag降/eff升/dur升/std降/jack降)"""
import os, sys, io, pickle, numpy as np, copy, itertools
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

def feats_of(path, lv_str):
    with open(path, 'rb') as f:
        cd, raw = load_chart_from_bytes(f.read())
    feats = extract_features(cd, speed=1.0)
    lv = 'AT' if 'AT' in lv_str.upper() else ('IN' if 'IN' in lv_str.upper() else 'HD')
    if lv == 'IN':
        for k, d in app_mod.DOMAIN_DELTA.items():
            if k in feats: feats[k] = feats[k] - d
    return feats

anchors = []
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
for r in cache['ranked']:
    if r['id'] == 7516: anchors.append(('Bathin', 17.2, r['feats'], r['level']))
    elif r['id'] == 59064: anchors.append(('ずんどこ', 15.8, r['feats'], r['level']))
    elif r['id'] == 15875: anchors.append(('FREEDOM DiVE', 16.15, r['feats'], r['level']))
anchors.append(('Apollo', 18.0, feats_of(os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '41242.json'), 'AT'), 'AT'))
anchors.append(('Chart_SP#1347', 17.65, feats_of(os.path.join(_ROOT, 'data', 'test_charts', 'Chart_SP #1347(1).json'), 'IN'), 'IN'))
TGT = np.array([a[1] for a in anchors])

def lv_key(s):
    s = (s or '').upper()
    if 'AT' in s: return 'AT'
    if 'IN' in s: return 'IN'
    if 'HD' in s: return 'HD'
    return 'IN'

def predict_full(feats_raw, level_str):
    feats = dict(feats_raw)
    lv = lv_key(level_str)
    if lv == 'IN':
        for k, d in app_mod.DOMAIN_DELTA.items():
            if k in feats: feats[k] = feats[k] - d
    lv2 = 'IN_AT' if lv in ('IN','AT') and 'IN_AT' in app_mod.LV_ORDER else lv
    if lv2 not in app_mod.LV_ORDER: lv2 = app_mod.LV_ORDER[-1]
    vec = [0.0]*len(app_mod.LV_ORDER); vec[app_mod.LV_ORDER.index(lv2)] = 1.0
    x = np.array([[feats.get(n,0) for n in app_mod.FN] + vec])
    p_gb = float(app_mod.gb.predict(app_mod.scaler.transform(x))[0])
    b, _, _ = app_mod.compute_boost(feats, 1.0, is_custom=True)
    pred = p_gb + b
    _H = {'叠键', '多押', '变速', '位移'}
    if 14 < pred <= 16.5 and sum(1 for t in app_mod.compute_tags(feats) if t in _H) >= 2:
        pred -= b * 0.08
    act = feats.get('tracks_active_sec', 0)
    if act > 0:
        pred += 0.15*min(feats.get('tracks_4plus_sec',0)/act,0.8) + 0.55*min(feats.get('tracks_5plus_sec',0)/act,0.4) + 1.0*min(feats.get('tracks_6plus_sec',0)/act,0.15)
    hr = feats.get('hold_count', 0)/max(feats.get('total_notes',1),1)
    if hr >= 0.6: pred += 0.7
    elif hr >= 0.4: pred += 0.5
    elif hr >= 0.25: pred += 0.3
    for lo, hi, adj in app_mod._CALIB_TABLE:
        if lo < pred <= hi: pred -= adj; break
    return pred

# 权重覆盖: {特征: 倍率}
FEATS = ['drag_per_sec', 'eff_peak_tps_1s', 'above_avg_duration_sec', 'density_transition_std', 'jack_max_run', 'long_jack_count', 'density_transition_mean', 'above_avg_density_mean']
results = []
for drag_m, eff_m, dur_m, std_m, jack_m in itertools.product([0.4, 0.6, 0.8, 1.0], [1.0, 1.2, 1.4], [1.0, 1.15, 1.3], [0.4, 0.7, 1.0], [0.4, 0.7, 1.0]):
    FLAT = copy.deepcopy(app_mod.MANUAL_FLAT)
    for i, (fname, bl, co) in enumerate(FLAT):
        if fname == 'drag_per_sec': FLAT[i] = (fname, bl, co*drag_m)
        elif fname == 'eff_peak_tps_1s': FLAT[i] = (fname, bl, co*eff_m)
        elif fname == 'above_avg_duration_sec': FLAT[i] = (fname, bl, co*dur_m)
        elif fname == 'density_transition_std': FLAT[i] = (fname, bl, co*std_m)
        elif fname == 'jack_max_run': FLAT[i] = (fname, bl, co*jack_m)
    saved = app_mod.MANUAL_FLAT
    app_mod.MANUAL_FLAT = FLAT
    ps = np.array([predict_full(a[2], a[3]) for a in anchors])
    app_mod.MANUAL_FLAT = saved
    errs = ps - TGT
    mae = np.abs(errs).mean()
    results.append((mae, drag_m, eff_m, dur_m, std_m, jack_m, errs))
results.sort(key=lambda x: x[0])
print('top10 组合 (MAE | drag eff dur std jack | 各锚点差):')
names = [a[0] for a in anchors]
for mae, d, e, du, s, j, errs in results[:10]:
    print(f'MAE={mae:.3f} drag={d} eff={e} dur={du} std={s} jack={j} | ' + ' '.join(f'{n[:6]}={x:+.2f}' for n, x in zip(names, errs)))
print('\nbaseline:')
FLAT = copy.deepcopy(app_mod.MANUAL_FLAT)
app_mod.MANUAL_FLAT = FLAT
ps = np.array([predict_full(a[2], a[3]) for a in anchors])
print(' '.join(f'{n}={x:+.2f}' for n, x in zip(names, ps - TGT)))
print('DONE')