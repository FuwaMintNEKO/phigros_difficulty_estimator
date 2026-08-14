# -*- coding: utf-8 -*-
"""宽维度随机搜索: 12特征权重, 5锚点MAE + 全量MAE"""
import os, sys, io, pickle, numpy as np, copy, random
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
TGT = np.array([a[1] for a in anchors]); names = [a[0] for a in anchors]
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])

grid = {
  'drag_per_sec': [0.2, 0.3, 0.5, 0.7, 1.0],
  'density_transition_std': [0.3, 0.5, 0.7, 1.0],
  'jack_max_run': [0.3, 0.5, 0.7, 1.0],
  'eff_peak_tps_1s': [1.0, 1.2, 1.4, 1.6],
  'above_avg_duration_sec': [1.0, 1.15, 1.3],
  'jline_movement_density': [0.5, 0.7, 1.0],
  'above_avg_density_mean': [0.8, 0.9, 1.0, 1.1],
  'weighted_mf_score_per_sec': [0.8, 0.9, 1.0, 1.1],
  'movement_per_second': [1.0, 1.2, 1.4],
  'type_switch_per_sec': [0.8, 1.0, 1.2],
  'chord_alternation_rate': [0.8, 1.0],
  'jline_rotate_density': [0.6, 0.8, 1.0],
}
keys = list(grid.keys())
best = []
random.seed(7)
for trial in range(500):
    ov = {k: random.choice(grid[k]) for k in keys}
    FLAT = copy.deepcopy(app_mod.MANUAL_FLAT)
    for i, (fname, bl, co) in enumerate(FLAT):
        if fname in ov: FLAT[i] = (fname, bl, co*ov[fname])
    saved = app_mod.MANUAL_FLAT
    app_mod.MANUAL_FLAT = FLAT
    ps_a = np.array([predict_full(a[2], a[3]) for a in anchors])
    errs_a = np.abs(ps_a - TGT).mean()
    ps = np.array([predict_full(r['feats'], r['level']) for r in ranked])
    errs = np.abs(ps - ds).mean()
    app_mod.MANUAL_FLAT = saved
    if errs <= 0.540:
        score = errs_a + (errs - 0.532) * 3.0
        best.append((score, errs_a, errs, ov, ps_a - TGT))
best.sort(key=lambda x: x[0])
print('top12:')
for sc, ea, em, ov, ea_all in best[:12]:
    print(f'score={sc:.3f} 锚MAE={ea:.3f} 全MAE={em:.3f}')
    print(f'   锚差: ' + ' '.join(f'{n[:6]}={x:+.2f}' for n, x in zip(names, ea_all)))
    print(f'   权重: ' + ' '.join(f'{k}={v}' for k, v in ov.items()))
print('DONE')