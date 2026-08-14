# -*- coding: utf-8 -*-
"""验证 Chart_SP #1347 (Spasmodic SP) 在 备份v10 vs 当前模型 下的 AT 档预测"""
import os, sys, pickle
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
import app as cur_app

fn = os.path.join(r'C:\Users\NaNK\Downloads', 'Chart_SP #1347(1).json')
with open(fn, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
feats = extract_features(cd)

with open(os.path.join(_ROOT, 'models', '6dim_model_v10_backup_old.pkl'), 'rb') as f:
    m = pickle.load(f)
gb_old, scaler_old, FN_old = m['gb'], m['scaler'], m['feature_names']
P95_old, P99_old, FLAT_old = m['p95_vals'], m['p99_vals'], m['MANUAL_FLAT']
CAPS_old = m.get('caps', {}) or {}
LV_ORDER = m.get('lv_order', ['EZ', 'HD', 'IN', 'AT'])

for level in ['EZ', 'HD', 'IN', 'AT']:
    x = np.array([[feats.get(n, 0) for n in FN_old] + [1.0 if level == l else 0.0 for l in LV_ORDER]])
    xs = scaler_old.transform(x)
    g = float(gb_old.predict(xs)[0])
    b = 0.0
    for fname, bl, co in FLAT_old:
        v = feats.get(fname, 0)
        pv = P95_old.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t: continue
        e = v / t - 1.0
        c = CAPS_old.get(fname, CAPS_old.get('_default', None))
        if c is not None and e > c: e = c
        x2 = co * (e ** 0.70)
        if v > max(P99_old.get(fname, 0), bl * 0.5):
            pe = v / max(P99_old.get(fname, 0), bl * 0.5) - 1.0
            if c is not None and pe > c: pe = c
            x2 += co * max(0, pe) ** 0.70 * 0.5
        b += x2
    p = g + b
    r_cur, _ = cur_app.predict_one_chart(cd, speed=1.0, level=level)
    p_cur = r_cur['prediction'] if r_cur else None
    print(f'{level}: 旧v10={p:.2f} (gb={g:.2f}+b={b:.2f})  当前={p_cur}')
