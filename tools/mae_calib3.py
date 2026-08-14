# -*- coding: utf-8 -*-
"""修正搜索: 预计算对齐feats (与生产完全一致)"""
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
print(f'评估集: {N} 首')

# 预计算: 对齐后的 feats (与生产 predict 完全一致)
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

# 先验证当前生产权重下的基线
BASE_CAL = app_mod._CALIB_TABLE
ps = np.array([predict_fast(i, BASE_CAL) for i in range(N)])
errs = ps - ds_f
print(f'生产基线(新权重+新校准): MAE={np.abs(errs).mean():.3f} bias={errs.mean():+.3f}')
for lo, hi, t2 in [(12,13,'12-13'),(13,14,'13-14'),(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
    mk = np.where((ds_f >= lo) & (ds_f < hi))[0]
    if len(mk): print(f'  {t2}: {errs[mk].mean():+.3f}')

# 校准扫描 (固定当前权重)
best = []
random.seed(9)
for trial in range(4000):
    calib = [(12,13,random.choice([-0.3,-0.2,-0.1,0.0])),
             (13,14,random.choice([-0.2,-0.15,-0.1,-0.05,0.0,0.05])),
             (14,15,random.choice([0.0,0.05,0.1,0.15,0.2])),
             (15,16,random.choice([0.05,0.1,0.15,0.2,0.25])),
             (16,16.5,random.choice([0.0,0.05,0.1,0.15,0.2])),
             (16.5,17,random.choice([0.0,0.05,0.1,0.15,0.2])),
             (17,99,random.choice([-0.1,-0.05,0.0,0.05,0.1]))]
    ps = np.array([predict_fast(i, calib) for i in range(N)])
    errs = ps - ds_f
    mae = np.abs(errs).mean()
    seg_bias = []
    for lo, hi in [(12,13),(13,14),(14,15),(15,16),(16,16.5),(16.5,17),(17,99)]:
        mk = np.where((ds_f >= lo) & (ds_f < hi))[0]
        if len(mk): seg_bias.append(abs(errs[mk].mean()))
    score = mae + 0.10*np.mean(seg_bias)
    best.append((score, mae, calib, errs, np.mean(seg_bias)))
best.sort(key=lambda x: x[0])
print('\ntop8 校准:')
for sc, mae, calib, errs, sb in best[:8]:
    segs = []
    for lo, hi, t2 in [(12,13,'12-13'),(13,14,'13-14'),(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
        mk = np.where((ds_f >= lo) & (ds_f < hi))[0]
        if len(mk): segs.append(f'{t2}:{errs[mk].mean():+.2f}')
    print(f'score={sc:.3f} MAE={mae:.3f} 段均偏={sb:.3f}')
    print(f'  校准={calib}')
    print(f'  段: {" ".join(segs)}')
print('DONE')