# -*- coding: utf-8 -*-
"""v11.9 vs v11.10 GB预测对比 (16.5+段)"""
import os, sys, numpy as np, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])
_ALIGN = app_mod.DOMAIN_DELTA

def load_model(path):
    with open(path, 'rb') as f:
        m = pickle.load(f)
    return m['gb'], m['scaler'], m['feature_names'], m['lv_order']

gb9, scl9, fn9, lv9 = load_model(os.path.join(_ROOT, 'models', '6dim_model_v11_9.pkl'))
gb10, scl10 = app_mod.gb, app_mod.scaler
FN10, LV10 = app_mod.FN, app_mod.LV_ORDER
print('fn9 len', len(fn9), 'fn10 len', len(FN10))
print('fn9-fn10:', sorted(set(fn9)-set(FN10)))
print('fn10-fn9:', sorted(set(FN10)-set(fn9)))
print('lv9:', lv9, ' lv10:', LV10)

def gb_pred(feats_raw, level, gb, scl, fn, lv_order):
    feats = dict(feats_raw)
    lv = level.upper()
    if 'AT' in lv: lv = 'AT'
    elif 'IN' in lv: lv = 'IN'
    elif 'HD' in lv: lv = 'HD'
    else: lv = 'IN'
    if lv == 'IN':
        for k, d in _ALIGN.items():
            if k in feats: feats[k] = feats[k] - d
    lv2 = 'IN_AT' if lv in ('IN','AT') and 'IN_AT' in lv_order else lv
    if lv2 not in lv_order: lv2 = lv_order[-1]
    vec = [0.0]*len(lv_order); vec[lv_order.index(lv2)] = 1.0
    x = np.array([[feats.get(n,0) for n in fn] + vec])
    return float(gb.predict(scl.transform(x))[0])

print('\n16.5+ 段 GB预测对比:')
print(f'{"谱名":<30}{"真实":>6}{"v9gb":>8}{"v10gb":>8}{"差":>7}')
for i, r in enumerate(ranked):
    if ds[i] >= 16.5:
        p9 = gb_pred(r['feats'], r['level'], gb9, scl9, fn9, lv9)
        p10 = gb_pred(r['feats'], r['level'], gb10, scl10, FN10, LV10)
        print(f'{r["name"][:30]:<30}{ds[i]:>6.2f}{p9:>8.3f}{p10:>8.3f}{p10-p9:>+7.3f}')
print('DONE')