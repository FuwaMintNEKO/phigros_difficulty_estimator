# -*- coding: utf-8 -*-
"""锚点谱详细特征贡献对比 (修正level解析)"""
import os, sys, io, pickle, numpy as np
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

def full(feats_raw, level_str):
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
    b, dims, kf = app_mod.compute_boost(feats, 1.0, is_custom=True)
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
    return pred, p_gb, b, dims, kf

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
for r in cache['ranked']:
    if r['id'] in (7516, 59064, 15875):
        pred, p_gb, b, dims, kf = full(r['feats'], r['level'])
        print(f"=== {r['name'][:22]} id={r['id']} diff={round(r['diff'],1)} pred={pred:.2f} ===")
        print(f"  gb={p_gb:.3f} boost={b:.3f} cats={ {k: round(float(v),3) for k,v in dims['categories'].items()} }")
        print('  顶贡献:', [(c[0], round(c[1],3)) for c in kf[:6]])
for cid, path, lv, nm in [
    (41242, os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '41242.json'), 'AT', 'Apollo(未上架)'),
    (None, os.path.join(_ROOT, 'data', 'test_charts', 'Chart_SP #1347(1).json'), 'IN', 'Chart_SP#1347'),
]:
    with open(path, 'rb') as f:
        cd, raw = load_chart_from_bytes(f.read())
    feats = extract_features(cd, speed=1.0)
    pred, p_gb, b, dims, kf = full(feats, lv)
    print(f"=== {nm} pred={pred:.2f} ===")
    print(f"  gb={p_gb:.3f} boost={b:.3f} cats={ {k: round(float(v),3) for k,v in dims['categories'].items()} }")
    print('  顶贡献:', [(c[0], round(c[1],3)) for c in kf[:6]])
print('DONE')