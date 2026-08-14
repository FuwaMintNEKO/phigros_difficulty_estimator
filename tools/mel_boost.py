# -*- coding: utf-8 -*-
"""候选特征加入boost实验: miniburst/micro/tap_burst 对Melodiniq影响"""
import os, sys, io, json, pickle, numpy as np, copy
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

def predict_full(feats_raw, level_str, extra_boost=None):
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
    if extra_boost:
        for fname, bl, co in extra_boost:
            v = feats.get(fname, 0)
            t = max(app_mod.P95.get(fname, 0)*0.55, bl*0.5)
            if v > t:
                pred += co * ((v/t - 1.0) ** 0.7)
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

# Melodiniq 特征
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
feats_mel = extract_features(cd, speed=1.0)
print('Melodiniq 当前预测:', round(predict_full(feats_mel, 'IN'), 3))

# 候选特征: 24分/爆发类
cands = ['miniburst_count', 'miniburst_density', 'micro_max_0.0625beat', 'micro_peak_top5_0.0625beat',
         'tap_burst_05_top5', 'tap_burst_top5', 'tap_burst_peak_to_mean', 'tap_micro_top5_0.0625beat',
         'core_micro_top5_0.0625beat', 'fast_ms_050_ratio']
print('\n候选特征值 + P95:')
for f in cands:
    v = feats_mel.get(f, 0)
    p95 = app_mod.P95.get(f, 0)
    print(f'  {f:<32} v={v:.3f} P95={p95:.3f} 阈值={p95*0.55:.3f} 触发={v>p95*0.55}')
# 实验: 各特征加入boost
print('\n各特征独立加入boost (co=0.1) 对 Melodiniq 的影响:')
for f in cands:
    p2 = predict_full(feats_mel, 'IN', [(f, 1.0, 0.10)])
    print(f'  {f:<32} → {p2:.3f} (+{p2-16.234:.3f})')
# 组合: miniburst_density + tap_burst_top5 + micro_max
combos = [
    ('miniburst_density 0.15', [('miniburst_density', 0.02, 0.15)]),
    ('miniburst_count 0.15', [('miniburst_count', 200.0, 0.15)]),
    ('micro_max_0.0625beat 0.3', [('micro_max_0.0625beat', 2.0, 0.30)]),
    ('tap_burst_top5 0.3', [('tap_burst_top5', 0.5, 0.30)]),
    ('全部组合', [('miniburst_density', 0.02, 0.15), ('micro_max_0.0625beat', 2.0, 0.20), ('tap_burst_top5', 0.5, 0.15)]),
]
print('\n组合实验:')
for tag, extra in combos:
    p2 = predict_full(feats_mel, 'IN', extra)
    print(f'  {tag:<36} → {p2:.3f} (+{p2-16.234:.3f})')
print('DONE')