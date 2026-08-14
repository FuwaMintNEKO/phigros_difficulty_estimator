# -*- coding: utf-8 -*-
"""锚点最优组合 + ranked全量MAE验证"""
import os, sys, io, pickle, numpy as np, copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

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

# 候选组合
combos = [
    ('baseline', {}),
    ('drag.4/eff1.0/dur1.3/std.4/jack.4', {'drag_per_sec':0.4, 'above_avg_duration_sec':1.3, 'density_transition_std':0.4, 'jack_max_run':0.4}),
    ('drag.4/eff1.4/dur1.15/std.4/jack.4', {'drag_per_sec':0.4, 'eff_peak_tps_1s':1.4, 'above_avg_duration_sec':1.15, 'density_transition_std':0.4, 'jack_max_run':0.4}),
    ('drag.5/eff1.3/dur1.2/std.5/jack.5', {'drag_per_sec':0.5, 'eff_peak_tps_1s':1.3, 'above_avg_duration_sec':1.2, 'density_transition_std':0.5, 'jack_max_run':0.5}),
]
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])
for tag, ov in combos:
    FLAT = copy.deepcopy(app_mod.MANUAL_FLAT)
    for i, (fname, bl, co) in enumerate(FLAT):
        if fname in ov: FLAT[i] = (fname, bl, co*ov[fname])
    saved = app_mod.MANUAL_FLAT
    app_mod.MANUAL_FLAT = FLAT
    ps = np.array([predict_full(r['feats'], r['level']) for r in ranked])
    errs = ps - ds
    segs = []
    for lo, hi, t2 in [(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
        mk = np.where((ds >= lo) & (ds < hi))[0]
        if len(mk): segs.append(f'{t2}:{errs[mk].mean():+.3f}')
    print(f'{tag:<34} MAE={np.abs(errs).mean():.3f} bias={errs.mean():+.3f} | {" ".join(segs)}')
    app_mod.MANUAL_FLAT = saved
print('DONE')