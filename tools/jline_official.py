# -*- coding: utf-8 -*-
"""jline P95修正对官谱的影响 (官谱是唯一权威)"""
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

def predict_full(feats_raw, level_str, is_custom):
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
    b, _, _ = app_mod.compute_boost(feats, 1.0, is_custom=is_custom)
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
# 官谱判定: 用 kyou_type_for 文件名匹配 (kyou_type_for(feats, name, is_custom) 官谱返回共识类型)
off_idx = []
for i, r in enumerate(ranked):
    kt = app_mod.kyou_type_for(r['feats'], r['name'], True)
    if kt and kt.get('type'):
        off_idx.append(i)
off_idx = np.array(off_idx)
print(f'官谱(kyou共识): {len(off_idx)}/{len(ranked)}')

# 新P95
def capped_p95(vals, cap=200):
    return np.percentile(np.minimum(vals, cap), 95)
new_p95 = {}
for k in ['jline_movement_density', 'jline_rotate_density', 'jline_disappear_density']:
    vals = np.array([r['feats'].get(k, 0) for r in ranked])
    new_p95[k] = capped_p95(vals, 200)

for tag, use_new in [('旧P95', False), ('新P95', True)]:
    if use_new:
        saved = app_mod.P95
        app_mod.P95 = dict(saved)
        for k, v in new_p95.items(): app_mod.P95[k] = v
    ps = np.array([predict_full(r['feats'], r['level'], True) for r in ranked])
    errs = ps - ds
    print(f'\n{tag} (自制谱管线):')
    print(f'  全量: MAE={np.abs(errs).mean():.3f} bias={errs.mean():+.3f}')
    print(f'  官谱段: 15-16:{errs[(ds>=15)&(ds<16)&np.isin(np.arange(len(ds)),off_idx)].mean():+.3f}(n={np.sum((ds>=15)&(ds<16)&np.isin(np.arange(len(ds)),off_idx))})')
    for lo, hi, t2 in [(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
        mk = np.where((ds >= lo) & (ds < hi))[0]
        off_mk = [i for i in mk if i in off_idx]
        if off_mk:
            print(f'  {t2}: 全部={errs[mk].mean():+.3f}(n={len(mk)}) 官谱={errs[off_mk].mean():+.3f}(n={len(off_mk)})')
    # 官谱boost变化
    if use_new:
        app_mod.P95 = saved
print('DONE')