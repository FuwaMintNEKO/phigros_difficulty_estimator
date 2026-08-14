# -*- coding: utf-8 -*-
"""Feeling Blue修复: ml加入boost + jline权重提升 + P95修正"""
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

def predict_full(feats_raw, level_str, extra=None, p95_override=None):
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
    if extra:
        P95X = p95_override or app_mod.P95
        for fname, bl, co in extra:
            v = feats.get(fname, 0)
            t = max(P95X.get(fname, 0)*0.55, bl*0.5)
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

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])

# 新P95 (cap=200)
def capped_p95(vals, cap=200):
    return np.percentile(np.minimum(vals, cap), 95)
p95n = {}
for k in ['jline_movement_density', 'jline_rotate_density', 'jline_disappear_density']:
    vals = np.array([r['feats'].get(k, 0) for r in ranked])
    p95n[k] = capped_p95(vals, 200)

# ml 的 P95 (模型里没有? 检查)
print('ml P95 in app:', app_mod.P95.get('multi_line_sim_events'))
ml_vals = np.array([r['feats'].get('multi_line_sim_events', 0) for r in ranked])
p95n['multi_line_sim_events'] = np.percentile(ml_vals, 95)
print('ml 新P95:', p95n['multi_line_sim_events'])

combos = [
    ('基线(v11.12权重)', None, None),
    ('ml加入boost 0.03', [('multi_line_sim_events', 30.0, 0.03)], None),
    ('ml 0.05', [('multi_line_sim_events', 30.0, 0.05)], None),
    ('ml 0.05 + jline权重修正P95', [('multi_line_sim_events', 30.0, 0.05)], p95n),
    ('ml 0.08 + jlineP95 + jrot 0.048x2', [('multi_line_sim_events', 30.0, 0.08), ('jline_rotate_density', 20.0, 0.048)], p95n),
]
for tag, extra, p95o in combos:
    ps = []
    for r in ranked:
        ps.append(predict_full(r['feats'], r['level'], extra, p95o))
    ps = np.array(ps)
    errs = ps - ds
    fb = None; bathin = None
    for i, r in enumerate(ranked):
        if r['id'] == 47264: fb = ps[i]
        if r['id'] == 7516: bathin = ps[i]
    segs = []
    for lo, hi, t2 in [(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
        mk = np.where((ds >= lo) & (ds < hi))[0]
        if len(mk): segs.append(f'{t2}:{errs[mk].mean():+.2f}')
    print(f'{tag:<30} MAE={np.abs(errs).mean():.3f} bias={errs.mean():+.3f} FB={fb:.2f} Bathin={bathin:.2f}')
    print(f'    {" ".join(segs)}')
print('DONE')