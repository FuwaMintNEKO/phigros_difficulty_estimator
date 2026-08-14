# -*- coding: utf-8 -*-
"""多面下落(ml)加入boost: 对Feeling Blue/锚点/全量影响"""
import os, sys, io, pickle, numpy as np, copy
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

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])

# ml 特征统计
ml = np.array([r['feats'].get('multi_line_sim_events', 0) for r in ranked])
print(f'ml: P50={np.median(ml):.0f} P90={np.percentile(ml,90):.0f} P95={np.percentile(ml,95):.0f} max={ml.max():.0f}')
# Feeling Blue 与类似ml谱的预测差
for r in ranked:
    if r['id'] == 47264:
        print(f'Feeling Blue: ml={r["feats"].get("multi_line_sim_events",0)} diff={round(r["diff"],1)}')
        # 用现有权重组合预测
        print('  baseline pred =', round(predict_full(r['feats'], r['level']), 2))

# 测试 ml boost 不同权重
for co_ml in [0.0, 0.02, 0.04, 0.08]:
    extra = [('multi_line_sim_events', 30.0, co_ml)]
    preds = []
    for r in ranked:
        preds.append(predict_full(r['feats'], r['level'], extra))
    ps = np.array(preds)
    errs = ps - ds
    # 锚点
    a_vals = []
    for r in ranked:
        if r['id'] == 7516: a_vals.append(('Bathin', 17.2, predict_full(r['feats'], r['level'], extra)))
        elif r['id'] == 59064: a_vals.append(('ずんどこ', 15.8, predict_full(r['feats'], r['level'], extra)))
        elif r['id'] == 15875: a_vals.append(('FREEDOM', 16.15, predict_full(r['feats'], r['level'], extra)))
    for r in ranked:
        if r['id'] == 47264:
            fb = predict_full(r['feats'], r['level'], extra)
    segs = []
    for lo, hi, t2 in [(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,99,'>=17')]:
        mk = np.where((ds >= lo) & (ds < hi))[0]
        if len(mk): segs.append(f'{t2}:{errs[mk].mean():+.2f}')
    print(f'ml co={co_ml}: 全MAE={np.abs(errs).mean():.3f} | {" ".join(segs)} | FB={fb:.2f} | ' + ' '.join(f'{n}={v:.2f}' for n, t, v in a_vals))
print('DONE')