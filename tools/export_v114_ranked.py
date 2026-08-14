# -*- coding: utf-8 -*-
"""上架589 全量预测导出 (v11.4 + 新校准) → v114_ranked_predictions.csv"""
import os, sys, io, csv, json, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import app as appmod

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
valid = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]

def fast_predict(feats_raw, level='IN'):
    feats = dict(feats_raw)
    lv = level.upper()
    if 'AT' in lv: lv = 'AT'
    elif 'IN' in lv: lv = 'IN'
    elif 'HD' in lv: lv = 'HD'
    else: lv = 'IN'
    if lv == 'IN':
        for k, d in appmod.DOMAIN_DELTA.items():
            if k in feats: feats[k] = feats[k] - d
    if 'IN_AT' in appmod.LV_ORDER and lv in ('IN','AT'): lv = 'IN_AT'
    if lv not in appmod.LV_ORDER: lv = appmod.LV_ORDER[-1]
    vec = [0.0]*len(appmod.LV_ORDER); vec[appmod.LV_ORDER.index(lv)] = 1.0
    x = np.array([[feats.get(n,0) for n in appmod.FN] + vec])
    xs = appmod.scaler.transform(x)
    p_gb = float(appmod.gb.predict(xs)[0])
    b0, _, _ = appmod.compute_boost(feats, speed=1.0, is_custom=True)
    pred = p_gb + b0
    act = feats.get('tracks_active_sec', 0)
    if act > 0:
        r4 = feats.get('tracks_4plus_sec', 0)/act; r5 = feats.get('tracks_5plus_sec', 0)/act; r6 = feats.get('tracks_6plus_sec', 0)/act
        pred += 0.15*min(r4,0.8) + 0.55*min(r5,0.4) + 1.0*min(r6,0.15)
    for lo, hi, adj in appmod._CALIB_TABLE:
        if lo < pred <= hi: pred -= adj; break
    return pred, p_gb, b0

rows = []
for r in valid:
    p, g, b = fast_predict(r['feats'], r['level'])
    f_ = r['feats']
    rows.append([r['id'], r['name'], r['level'], round(r['diff'], 2), round(p, 3), round(p - r['diff'], 3),
                 round(g, 3), round(b, 3),
                 f_.get('multi_finger_3plus_events', 0), f_.get('multi_finger_4plus_events', 0),
                 round(f_.get('above_avg_density_mean', 0), 2), round(f_.get('eff_avg_tps_1s', 0), 2),
                 round(f_.get('real_core_notes_per_second', 0), 2), f_.get('total_notes', 0),
                 round(f_.get('duration_sec', 0), 1)])

out = os.path.join(_ROOT, 'data', 'phira', 'v114_ranked_predictions.csv')
with open(out, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'name', 'level', 'diff', 'pred', 'err', 'gb', 'boost', 'mf3', 'mf4', 'dens', 'eff_avg', 'nps', 'notes', 'duration'])
    for r_ in rows:
        w.writerow(r_)
print(f'已写入: {out} ({len(rows)} 行)')
print('DONE')
