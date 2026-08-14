# -*- coding: utf-8 -*-
"""细校准表(6段) + 权重联合随机搜索: 目标MAE最小"""
import os, sys, io, json, pickle, numpy as np, csv, copy, random
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

def predict_full(feats_raw, level_str, calib):
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
    for lo, hi, adj in calib:
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
print(f'评估集: {len(ranked_f)} 首')

# 预计算 GB 部分 (加速)
GBS = []
for r in ranked_f:
    feats = dict(r['feats'])
    lv = lv_key(r['level'])
    if lv == 'IN':
        for k, d in app_mod.DOMAIN_DELTA.items():
            if k in feats: feats[k] = feats[k] - d
    lv2 = 'IN_AT' if lv in ('IN','AT') and 'IN_AT' in app_mod.LV_ORDER else lv
    if lv2 not in app_mod.LV_ORDER: lv2 = app_mod.LV_ORDER[-1]
    vec = [0.0]*len(app_mod.LV_ORDER); vec[app_mod.LV_ORDER.index(lv2)] = 1.0
    x = np.array([[feats.get(n,0) for n in app_mod.FN] + vec])
    GBS.append(float(app_mod.gb.predict(app_mod.scaler.transform(x))[0]))
GBS = np.array(GBS)

def predict_fast(idx, calib, FLAT):
    """用预计算GB + 当前FLAT"""
    feats = dict(ranked_f[idx]['feats'])
    lv = lv_key(ranked_f[idx]['level'])
    if lv == 'IN':
        for k, d in app_mod.DOMAIN_DELTA.items():
            if k in feats: feats[k] = feats[k] - d
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

# 权重候选 (v11.12基础上微调)
grid = {
  'drag_per_sec': [0.15, 0.2, 0.25, 0.3, 0.4],
  'density_transition_std': [0.5, 0.7, 1.0],
  'jack_max_run': [0.3, 0.4, 0.5, 0.7, 1.0],
  'eff_peak_tps_1s': [1.0, 1.1, 1.2, 1.3, 1.4],
  'above_avg_duration_sec': [1.0, 1.15, 1.25, 1.3, 1.4],
  'jline_movement_density': [0.4, 0.5, 0.7, 1.0],
  'above_avg_density_mean': [0.8, 0.9, 1.0],
  'weighted_mf_score_per_sec': [0.85, 0.9, 1.0],
  'movement_per_second': [1.0, 1.1, 1.2, 1.3],
  'type_switch_per_sec': [1.0, 1.2, 1.4],
  'chord_alternation_rate': [0.7, 0.8, 0.9, 1.0],
  'jline_rotate_density': [0.6, 0.8, 1.0],
}
keys = list(grid.keys())
BASE_FLAT = copy.deepcopy(app_mod.MANUAL_FLAT)
best = []
random.seed(42)
for trial in range(600):
    ov = {k: random.choice(grid[k]) for k in keys}
    FLAT = copy.deepcopy(BASE_FLAT)
    for i, (fname, bl, co) in enumerate(FLAT):
        if fname in ov: FLAT[i] = (fname, bl, co*ov[fname])
    saved = app_mod.MANUAL_FLAT
    app_mod.MANUAL_FLAT = FLAT
    # 校准表也随机: 5段 (13-14, 14-15, 15-16, 16-16.5, 16.5-17, 17+)
    a13 = random.choice([0.05, 0.10, 0.15])
    a14 = random.choice([0.20, 0.25, 0.30, 0.35])
    a15 = random.choice([0.15, 0.20, 0.25, 0.30])
    a16 = random.choice([0.10, 0.15, 0.20])
    a165 = random.choice([0.05, 0.10, 0.15])
    a17 = random.choice([0.0, 0.05, 0.10])
    calib = [(13,14,a13),(14,15,a14),(15,16,a15),(16,16.5,a16),(16.5,17,a165),(17,99,a17)]
    ps = np.array([predict_fast(i, calib, FLAT) for i in range(len(ranked_f))])
    errs = ps - ds_f
    mae = np.abs(errs).mean()
    # 段均衡惩罚
    seg_bias = []
    for lo, hi in [(13,14),(14,15),(15,16),(16,16.5),(16.5,17),(17,99)]:
        mk = np.where((ds_f >= lo) & (ds_f < hi))[0]
        if len(mk): seg_bias.append(abs(errs[mk].mean()))
    score = mae + 0.15*np.mean(seg_bias)
    best.append((score, mae, ov, calib, errs, np.mean(seg_bias)))
    app_mod.MANUAL_FLAT = saved
best.sort(key=lambda x: x[0])
print('\ntop8 (score | MAE | 权重 | 校准 | 段均偏):')
for sc, mae, ov, calib, errs, sb in best[:8]:
    segs = []
    for lo, hi, t2 in [(13,14,'13-14'),(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
        mk = np.where((ds_f >= lo) & (ds_f < hi))[0]
        if len(mk): segs.append(f'{t2}:{errs[mk].mean():+.2f}')
    print(f'score={sc:.3f} MAE={mae:.3f} 段均偏={sb:.3f} | {" ".join(segs)}')
    print(f'  校准={calib}')
    print(f'  权重=' + ' '.join(f'{k}={v}' for k, v in ov.items()))
print('DONE')