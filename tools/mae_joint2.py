# -*- coding: utf-8 -*-
"""对齐生产: 权重+校准联合搜索 (v11.13权重为基础)"""
import os, sys, io, json, pickle, numpy as np, copy, random
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

charts = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
up_ids = {c['id'] for c in charts['上架']}
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10 and r['id'] in up_ids]
ds = np.array([round(r['diff'],1) for r in ranked])
int_mask = np.abs(ds - np.round(ds)) < 1e-6
ranked_f = [r for i, r in enumerate(ranked) if not int_mask[i]]
ds_f = ds[~int_mask]
N = len(ranked_f)

ALIGN_FEATS = []
GBS = []
for r in ranked_f:
    feats = dict(r['feats'])
    lv = lv_key(r['level'])
    if lv == 'IN':
        for k, d in app_mod.DOMAIN_DELTA.items():
            if k in feats: feats[k] = feats[k] - d
    ALIGN_FEATS.append(feats)
    lv2 = 'IN_AT' if lv in ('IN','AT') and 'IN_AT' in app_mod.LV_ORDER else lv
    if lv2 not in app_mod.LV_ORDER: lv2 = app_mod.LV_ORDER[-1]
    vec = [0.0]*len(app_mod.LV_ORDER); vec[app_mod.LV_ORDER.index(lv2)] = 1.0
    x = np.array([[feats.get(n,0) for n in app_mod.FN] + vec])
    GBS.append(float(app_mod.gb.predict(app_mod.scaler.transform(x))[0]))
GBS = np.array(GBS)

def predict_fast(idx, calib):
    feats = ALIGN_FEATS[idx]
    b, _, _ = app_mod.compute_boost(feats, 1.0, is_custom=True)
    pred = GBS[idx] + b
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
    for lo, hi, adj in calib:
        if lo < pred <= hi: pred -= adj; break
    return pred

# 权重网格 (当前v11.13权重为基础)
grid = {
  'drag_per_sec': [0.2, 0.3, 0.4, 0.6],
  'density_transition_std': [0.5, 0.7, 1.0],
  'jack_max_run': [0.3, 0.5, 0.7, 1.0],
  'eff_peak_tps_1s': [0.9, 1.0, 1.1, 1.2],
  'above_avg_duration_sec': [1.0, 1.15, 1.25, 1.4],
  'jline_movement_density': [0.3, 0.4, 0.5, 0.7],
  'above_avg_density_mean': [0.9, 1.0, 1.1],
  'weighted_mf_score_per_sec': [0.8, 0.85, 0.9, 1.0],
  'movement_per_second': [1.0, 1.1, 1.2],
  'type_switch_per_sec': [1.0, 1.2],
  'chord_alternation_rate': [0.6, 0.7, 0.8, 1.0],
  'jline_rotate_density': [0.6, 0.8, 1.0],
  'flash_hold_ratio': [0.8, 1.0, 1.2],
  'hold_interference_index': [0.8, 1.0, 1.2],
  'cross_hand_density': [0.8, 1.0, 1.2],
}
keys = list(grid.keys())
BASE_FLAT = copy.deepcopy(app_mod.MANUAL_FLAT)
best_cal = [(12,13,-0.3),(13,14,-0.15),(14,15,0.05),(15,16,0.1),(16,16.5,0.1),(16.5,17,0.2),(17,99,0.05)]
best = []
random.seed(33)
for trial in range(800):
    ov = {k: random.choice(grid[k]) for k in keys}
    FLAT = copy.deepcopy(BASE_FLAT)
    for i, (fname, bl, co) in enumerate(FLAT):
        if fname in ov: FLAT[i] = (fname, bl, co*ov[fname])
    saved = app_mod.MANUAL_FLAT
    app_mod.MANUAL_FLAT = FLAT
    ps = np.array([predict_fast(i, best_cal) for i in range(N)])
    errs = ps - ds_f
    mae = np.abs(errs).mean()
    seg_bias = []
    for lo, hi in [(12,13),(13,14),(14,15),(15,16),(16,16.5),(16.5,17),(17,99)]:
        mk = np.where((ds_f >= lo) & (ds_f < hi))[0]
        if len(mk): seg_bias.append(abs(errs[mk].mean()))
    score = mae + 0.08*np.mean(seg_bias)
    best.append((score, mae, ov, errs, np.mean(seg_bias)))
    app_mod.MANUAL_FLAT = saved
best.sort(key=lambda x: x[0])
print('top8 (权重搜索, 固定最优校准):')
for sc, mae, ov, errs, sb in best[:8]:
    segs = []
    for lo, hi, t2 in [(12,13,'12-13'),(13,14,'13-14'),(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
        mk = np.where((ds_f >= lo) & (ds_f < hi))[0]
        if len(mk): segs.append(f'{t2}:{errs[mk].mean():+.2f}')
    print(f'score={sc:.3f} MAE={mae:.3f} 段均偏={sb:.3f}')
    print(f'  权重=' + ' '.join(f'{k}={v}' for k, v in ov.items()))
    print(f'  段: {" ".join(segs)}')
print('DONE')