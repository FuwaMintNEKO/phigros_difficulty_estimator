# -*- coding: utf-8 -*-
"""v11.12生产验证: 锚点 + 全量 + 用户实测谱"""
import os, sys, io, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import importlib
import app as app_mod
importlib.reload(app_mod)
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

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])

print('=== 锚点 (v11.12生产) ===')
for r in ranked:
    if r['id'] == 7516: print(f"  Bathin: 锚=17.2 预测={predict_full(r['feats'], r['level']):.2f}")
    elif r['id'] == 59064: print(f"  ずんどこ: 锚=15.8 预测={predict_full(r['feats'], r['level']):.2f}")
    elif r['id'] == 15875: print(f"  FREEDOM DiVE: 锚=16.15 预测={predict_full(r['feats'], r['level']):.2f}")
fb = feats_of(os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '41242.json'), 'AT')
print(f"  Apollo: 锚=18.0 预测={predict_full(fb, 'AT'):.2f}")
cs = feats_of(os.path.join(_ROOT, 'data', 'test_charts', 'Chart_SP #1347(1).json'), 'IN')
print(f"  Chart_SP#1347: 锚=17.65 预测={predict_full(cs, 'IN'):.2f}")
print('\n=== 用户实测谱 ===')
for r in ranked:
    if r['id'] in (47264, 21041, 580, 8706, 28438):
        print(f"  {r['name'][:22]:<24} id={r['id']} diff={round(r['diff'],1)} 预测={predict_full(r['feats'], r['level']):.2f}")
print('\n=== 全量 ===')
ps = np.array([predict_full(r['feats'], r['level']) for r in ranked])
errs = ps - ds
segs = []
for lo, hi, t2 in [(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
    mk = np.where((ds >= lo) & (ds < hi))[0]
    if len(mk): segs.append(f'{t2}:{errs[mk].mean():+.3f}(n={len(mk)})')
print(f'MAE={np.abs(errs).mean():.3f} bias={errs.mean():+.3f}')
print('  ' + ' '.join(segs))
print('DONE')