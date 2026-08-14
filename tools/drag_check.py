# -*- coding: utf-8 -*-
"""drag_per_sec 与预测误差的关系 (验证drag降权是否系统合理)"""
import os, sys, io, pickle, numpy as np
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

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])
ps = np.array([predict_full(r['feats'], r['level']) for r in ranked])
errs = ps - ds

print('=== 按 drag_per_sec 分桶 (ranked全部) ===')
dr = np.array([r['feats'].get('drag_per_sec', 0) for r in ranked])
for lo, hi in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,9),(9,99)]:
    mk = np.where((dr >= lo) & (dr < hi))[0]
    if len(mk):
        print(f'drag[{lo},{hi}): n={len(mk)} bias={errs[mk].mean():+.3f} MAE={np.abs(errs[mk]).mean():.3f}')
print()
print('=== 高drag谱 (drag>=4) 逐谱 ===')
for i in np.where(dr >= 4)[0]:
    print(f'  {ranked[i]["name"][:26]:<28} drag={dr[i]:.2f} diff={ds[i]:.2f} pred={ps[i]:.2f} err={errs[i]:+.2f}')
print()
print('=== 官谱(kyou共识)中 drag 分布 ===')
# 用 kyou_type_for 判断官谱? 直接打印 drag 最高15首
idx = np.argsort(-dr)[:15]
for i in idx:
    print(f'  {ranked[i]["name"][:26]:<28} drag={dr[i]:.2f} diff={ds[i]:.2f} err={errs[i]:+.2f}')
print('DONE')