# -*- coding: utf-8 -*-
"""v11.13 生产验证: app.py 实际管线 (含P95修正+新权重+新校准)"""
import os, sys, io, json, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod

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

charts = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
up_ids = {c['id'] for c in charts['上架']}
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10 and r['id'] in up_ids]
ds = np.array([round(r['diff'],1) for r in ranked])
int_mask = np.abs(ds - np.round(ds)) < 1e-6
ranked_f = [r for i, r in enumerate(ranked) if not int_mask[i]]
ds_f = ds[~int_mask]

ps = np.array([predict_full(r['feats'], r['level']) for r in ranked_f])
errs = ps - ds_f
from scipy.stats import spearmanr
print(f'v11.13 过滤集 n={len(ps_f := ps)} MAE={np.abs(errs).mean():.3f} RMSE={np.sqrt((errs**2).mean()):.3f} bias={errs.mean():+.3f} rho={spearmanr(ps, ds_f).statistic:.3f}')
segs = []
for lo, hi, t2 in [(12,13,'12-13'),(13,14,'13-14'),(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
    mk = np.where((ds_f >= lo) & (ds_f < hi))[0]
    if len(mk): segs.append(f'{t2}:{errs[mk].mean():+.3f}(n={len(mk)})')
print('  ' + ' '.join(segs))
# 锚点
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
print('\n锚点:')
for r in ranked:
    if r['id'] == 7516: print(f"  Bathin: 锚=17.2 预测={predict_full(r['feats'], r['level']):.2f}")
    elif r['id'] == 59064: print(f"  ずんどこ: 锚=15.8 预测={predict_full(r['feats'], r['level']):.2f}")
    elif r['id'] == 15875: print(f"  FREEDOM DiVE: 锚=16.15 预测={predict_full(r['feats'], r['level']):.2f}")
print('app.VERSION =', app_mod.VERSION)
print('DONE')