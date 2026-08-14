# -*- coding: utf-8 -*-
"""验证CSV与生产管线一致 + 段统计"""
import os, sys, numpy as np, io, pickle, csv
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

def predict_app(feats_raw, level):
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
    b, _, _ = app_mod.compute_boost(feats, 1.0, is_custom=True)
    pred = p_gb + b
    _HIGH_TAGS = {'叠键', '多押', '变速', '位移'}
    if 14 < pred <= 16.5 and sum(1 for t in app_mod.compute_tags(feats) if t in _HIGH_TAGS) >= 2:
        pred -= b * 0.08
    act = feats_raw.get('tracks_active_sec', 0)
    if act > 0:
        pred += 0.15*min(feats_raw.get('tracks_4plus_sec',0)/act,0.8) + 0.55*min(feats_raw.get('tracks_5plus_sec',0)/act,0.4) + 1.0*min(feats_raw.get('tracks_6plus_sec',0)/act,0.15)
    hr = feats_raw.get('hold_count', 0) / max(feats_raw.get('total_notes', 1), 1)
    if hr >= 0.6: pred += 0.7
    elif hr >= 0.4: pred += 0.5
    elif hr >= 0.25: pred += 0.3
    for lo, hi, adj in app_mod._CALIB_TABLE:
        if lo < pred <= hi: pred -= adj; break
    return pred

ps = np.array([predict_app(r['feats'], r['level']) for r in ranked])
errs = ps - ds
# CSV一致性
with open(os.path.join(_ROOT, 'data', 'phira', 'v1111_ranked_predictions.csv'), encoding='utf-8-sig') as f:
    rows = list(csv.reader(f))[1:]
csv_pred = np.array([float(x[4]) for x in rows])
print(f'CSV一致性: maxdiff={np.abs(csv_pred-ps).max():.4f}')
segs = []
for lo, hi, t2 in [(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
    mk = np.where((ds >= lo) & (ds < hi))[0]
    if len(mk): segs.append(f'{t2}:{errs[mk].mean():+.3f}(n={len(mk)})')
print(f'v11.11生产 MAE={np.abs(errs).mean():.3f} bias={errs.mean():+.3f}')
print('  ' + ' '.join(segs))
print('DONE')