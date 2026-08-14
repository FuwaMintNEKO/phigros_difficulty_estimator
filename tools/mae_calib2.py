# -*- coding: utf-8 -*-
"""允许负校准adj: 修复13-14低估段, 联合权重搜索"""
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

# 预计算 GB
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

def predict_fast(idx, calib):
    feats = dict(ranked_f[idx]['feats'])
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

# 权重网格 (固定当前v11.12权重, 只搜校准) — 先只搜校准(含负adj)
calib_grid = []
for a12 in [-0.2,-0.1,0.0,0.1]:
  for a13 in [-0.15,-0.1,-0.05,0.0,0.05,0.1]:
    for a14 in [0.1,0.15,0.2,0.25,0.3,0.35]:
      for a15 in [0.1,0.15,0.2,0.25,0.3]:
        for a16 in [0.0,0.05,0.1,0.15,0.2]:
          for a165 in [0.0,0.05,0.1,0.15,0.2]:
            for a17 in [-0.1,-0.05,0.0,0.05,0.1]:
              calib_grid.append([(12,13,a12),(13,14,a13),(14,15,a14),(15,16,a15),(16,16.5,a16),(16.5,17,a165),(17,99,a17)])
print('校准组合数:', len(calib_grid))
random.seed(5)
random.shuffle(calib_grid)
# 只评估 3000 个随机校准 (每组合 410 次预测, 要控制时间)
best = []
for ci, calib in enumerate(calib_grid[:2500]):
    ps = np.array([predict_fast(i, calib) for i in range(len(ranked_f))])
    errs = ps - ds_f
    mae = np.abs(errs).mean()
    seg_bias = []
    for lo, hi in [(12,13),(13,14),(14,15),(15,16),(16,16.5),(16.5,17),(17,99)]:
        mk = np.where((ds_f >= lo) & (ds_f < hi))[0]
        if len(mk): seg_bias.append(abs(errs[mk].mean()))
    score = mae + 0.12*np.mean(seg_bias)
    best.append((score, mae, calib, errs, np.mean(seg_bias)))
best.sort(key=lambda x: x[0])
print('\ntop10 校准 (score | MAE | 段均偏):')
for sc, mae, calib, errs, sb in best[:10]:
    segs = []
    for lo, hi, t2 in [(12,13,'12-13'),(13,14,'13-14'),(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
        mk = np.where((ds_f >= lo) & (ds_f < hi))[0]
        if len(mk): segs.append(f'{t2}:{errs[mk].mean():+.2f}')
    print(f'score={sc:.3f} MAE={mae:.3f} 段均偏={sb:.3f}')
    print(f'  校准={calib}')
    print(f'  段: {" ".join(segs)}')
print('DONE')