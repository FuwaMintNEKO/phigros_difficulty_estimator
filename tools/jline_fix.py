# -*- coding: utf-8 -*-
"""修正jline P95 (剔除瞬移污染): 对Feeling Blue/全量影响"""
import os, sys, io, pickle, numpy as np, copy
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

# 新P95: 用ranked P90 (剔除瞬移异常) 或 cap 后计算
# 计算 cap 后 P95: 值 > 200 视为 200
def capped_p95(vals, cap):
    v = np.minimum(vals, cap)
    return np.percentile(v, 95)

new_p95 = {}
for k in ['jline_movement_density', 'jline_rotate_density', 'jline_disappear_density']:
    vals = np.array([r['feats'].get(k, 0) for r in ranked])
    new_p95[k] = capped_p95(vals, 200)
    print(f'{k}: 旧P95={app_mod.P95.get(k,0):.1f} → 新P95={new_p95[k]:.1f}')

# 应用新P95
saved_p95 = app_mod.P95
app_mod.P95 = dict(saved_p95)
for k, v in new_p95.items():
    app_mod.P95[k] = v

ps = np.array([predict_full(r['feats'], r['level']) for r in ranked])
errs = ps - ds
segs = []
for lo, hi, t2 in [(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
    mk = np.where((ds >= lo) & (ds < hi))[0]
    if len(mk): segs.append(f'{t2}:{errs[mk].mean():+.3f}')
print(f'\n新P95: MAE={np.abs(errs).mean():.3f} bias={errs.mean():+.3f}')
print('  ' + ' '.join(segs))
for r in ranked:
    if r['id'] in (47264, 7516):
        p = predict_full(r['feats'], r['level'])
        print(f'  {r["name"][:16]} id={r["id"]} diff={round(r["diff"],1)} 预测={p:.2f}')
app_mod.P95 = saved_p95
print('DONE')