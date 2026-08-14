# -*- coding: utf-8 -*-
"""5锚点预测表 + 权重敏感性分析"""
import os, sys, io, pickle, numpy as np, copy
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

def lv_key(s):
    s = (s or '').upper()
    if 'AT' in s: return 'AT'
    if 'IN' in s: return 'IN'
    if 'HD' in s: return 'HD'
    return 'IN'

def predict_full(feats_raw, level_str, calib_override=None):
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
    calib = calib_override or app_mod._CALIB_TABLE
    for lo, hi, adj in calib:
        if lo < pred <= hi: pred -= adj; break
    return pred

print('=== 当前预测 vs 锚点 ===')
base = []
for nm, tgt, feats, lv in anchors:
    p = predict_full(feats, lv)
    base.append(p)
    print(f'{nm:<16} 锚点={tgt:>6.2f} 当前={p:>6.2f} 差={p-tgt:>+6.2f}')

# 敏感性: 逐个特征权重 ±20%
print('\n=== 敏感性 (权重±20% 对锚点差的影响) ===')
print(f'{"特征":<32}' + ''.join(f'{n[:8]:>10}' for n, _, _, _ in anchors))
FLAT = list(app_mod.MANUAL_FLAT)
for idx, (fname, bl, co) in enumerate(FLAT):
    for delta in (0.8, 1.2):
        # 修改 app_mod.MANUAL_FLAT 副本
        FLAT2 = copy.deepcopy(FLAT)
        FLAT2[idx] = (fname, bl, co * delta)
        saved = app_mod.MANUAL_FLAT
        app_mod.MANUAL_FLAT = FLAT2
        deltas = []
        for i, (nm, tgt, feats, lv) in enumerate(anchors):
            p = predict_full(feats, lv)
            deltas.append(round(p - base[i], 3))
        app_mod.MANUAL_FLAT = saved
        print(f'{fname + (" x0.8" if delta==0.8 else " x1.2"):<32}' + ''.join(f'{d:>10.3f}' for d in deltas))
print('DONE')