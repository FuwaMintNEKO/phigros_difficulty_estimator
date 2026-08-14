# -*- coding: utf-8 -*-
"""对比: 备份原始v10模型 vs 当前模型 对自制谱定数的预测偏差
用备份 pkl 直接预测 (不覆盖当前模型), 输出两者对照
"""
import os, sys, re, pickle
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
import app as cur_app  # 当前模型

DL = r'C:\Users\NaNK\Downloads'
PAT = re.compile(r'^(.*?)\((\d+(?:\.\d+)?)(?:~(\d+(?:\.\d+)?))?\)(?:\(\d+\))?[^.]*\.json$')

def level_for(d):
    if d is None: return 'AT'
    if d >= 16.5: return 'AT'
    if d >= 11.5: return 'IN'
    if d >= 6.5: return 'HD'
    return 'EZ'

# 加载备份模型
with open(os.path.join(_ROOT, 'models', '6dim_model_v10_backup_old.pkl'), 'rb') as f:
    m = pickle.load(f)
gb_old, scaler_old, FN_old = m['gb'], m['scaler'], m['feature_names']
P95_old, P99_old, FLAT_old = m['p95_vals'], m['p99_vals'], m['MANUAL_FLAT']
CAPS_old = m.get('caps', {}) or {}
LV_ORDER = m.get('lv_order', ['EZ', 'HD', 'IN', 'AT'])

def predict_backup(feats, level):
    x = np.array([[feats.get(n, 0) for n in FN_old] + [1.0 if level == l else 0.0 for l in LV_ORDER]])
    xs = scaler_old.transform(x)
    g = float(gb_old.predict(xs)[0])
    # boost (同 app.compute_boost 逻辑, 用备份模型的 MANUAL_FLAT/caps)
    b = 0.0
    cap_default = CAPS_old.get('_default', None)
    for fname, bl, co in FLAT_old:
        v = feats.get(fname, 0)
        pv = P95_old.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t: continue
        e = v / t - 1.0
        c = CAPS_old.get(fname, cap_default)
        if c is not None and e > c: e = c
        x2 = co * (e ** 0.70)
        if v > max(P99_old.get(fname, 0), bl * 0.5):
            pe = v / max(P99_old.get(fname, 0), bl * 0.5) - 1.0
            if c is not None and pe > c: pe = c
            x2 += co * max(0, pe) ** 0.70 * 0.5
        b += x2
    return g + b, g, b

rows = []
for fn in sorted(os.listdir(DL)):
    if not fn.lower().endswith('.json'): continue
    mm = PAT.match(fn)
    if not mm: continue
    name, a, b2 = mm.group(1), float(mm.group(2)), mm.group(3)
    ud = (a + float(b2)) / 2 if b2 else a
    try:
        with open(os.path.join(DL, fn), 'rb') as f:
            cd, _ = load_chart_from_bytes(f.read())
        feats = extract_features(cd)
        lv = level_for(ud)
        p_old, g_old, b_old = predict_backup(feats, lv)
        res_cur, err_cur = cur_app.predict_one_chart(cd, speed=1.0, level=lv)
        p_cur = res_cur['prediction'] if res_cur else float('nan')
        rows.append((name, ud, lv, p_old, p_cur))
    except Exception as e:
        print(f'ERR {fn}: {e}')

print(f'{"谱面":<26} {"定数":>6} {"旧v10":>7} {"现":>7} {"旧偏差":>7} {"现偏差":>7}')
print('-' * 78)
for name, ud, lv, p_old, p_cur in sorted(rows, key=lambda r: -r[1]):
    print(f'{str(name)[:26]:<26} {ud:>6.1f} {p_old:>7.2f} {p_cur:>7.2f} {p_old-ud:>+7.2f} {p_cur-ud:>+7.2f}')

labeled = rows
ext = [r for r in labeled if r[1] > 17.5]
intr = [r for r in labeled if r[1] <= 17.5]
def mae(rs, idx):
    return sum(abs(r[idx] - r[1]) for r in rs) / len(rs) if rs else float('nan')
print(f'\n旧v10: 外推MAE={mae(ext,3):.3f} (n={len(ext)})  内推MAE={mae(intr,3):.3f} (n={len(intr)})')
print(f'当前: 外推MAE={mae(ext,4):.3f} (n={len(ext)})  内推MAE={mae(intr,4):.3f} (n={len(intr)})')
