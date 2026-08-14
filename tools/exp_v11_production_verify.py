# -*- coding: utf-8 -*-
"""v11 综合验证: app.py 生产路径
1. 官谱982 in-sample (应保持原精度, 条件boost仅自制谱)
2. 上架谱589 (条件boost+校准生效)
3. test_charts 17张 (用户可见效果)
"""
import os, sys, json, pickle, csv as _csv, numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import app as appmod
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json

print(f'生产模型: {appmod.VERSION}')
print(f'lv_order: {appmod.LV_ORDER}')

# ===== 1. 官谱 in-sample =====
song_difficulties = load_difficulty_tsv(os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv'))
chart_files = find_chart_files(os.path.join(_ROOT, 'data', 'chart'))
errs = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    for lv in ['EZ','HD','IN','AT']:
        if lv in info['levels'] and lv in diffs:
            try:
                cd = load_chart_json(info['levels'][lv])
                r, e = appmod.predict_one_chart(cd, level=lv, is_custom=False)
                if r: errs.append(r['prediction'] - diffs[lv])
            except Exception:
                pass
errs = np.array(errs)
print(f'\n官谱 in-sample: n={len(errs)} MAE={np.abs(errs).mean():.4f} bias={errs.mean():+.4f}')

# ===== 2. 上架谱 =====
charts = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
meta_by_id = {}
for lst in charts.values():
    for c in lst:
        meta_by_id[c['id']] = c
def read_csv_cols(path):
    rows = {}
    if not os.path.exists(path): return rows
    with open(path, encoding='utf-8-sig', newline='') as f:
        rd = _csv.reader(f)
        head = next(rd)
        for c in rd:
            if len(c) < len(head): continue
            o = dict(zip(head, c))
            try: rows[int(o['id'])] = o
            except Exception: pass
    return rows
pred_old = read_csv_cols(os.path.join(_ROOT, 'data', 'phira', 'predictions.csv'))

JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json')
results = []
for fn in sorted(os.listdir(JSON_DIR)):
    if not fn.endswith('.json'): continue
    cid = int(fn[:-5])
    meta = meta_by_id.get(cid, {})
    old = pred_old.get(cid, {})
    diff = old.get('diff')
    if not diff or float(diff) <= 10: continue
    try:
        with open(os.path.join(JSON_DIR, fn), 'rb') as f:
            chart_data, raw_text = appmod.load_chart_from_bytes(f.read())
        if chart_data is None: continue
        r, e = appmod.predict_one_chart(chart_data, level=meta.get('level', 'IN'), is_custom=True)
        if r: results.append({'diff': float(diff), 'pred': r['prediction'], 'name': meta.get('name','')})
    except Exception:
        pass
print(f'\n上架谱 (生产路径, 条件boost+校准): n={len(results)}')
bins = {}
for r in results:
    d = r['diff']
    bin_ = d < 13 and '<13' or d < 14 and '13-14' or d < 15 and '14-15' or d < 16 and '15-16' or d < 17 and '16-17' or '>=17'
    b = bins.setdefault(bin_, {'n':0,'b':0,'mae':0})
    b['n'] += 1; b['b'] += r['pred']-d; b['mae'] += abs(r['pred']-d)
for k in sorted(bins, key=lambda x: float(x.replace('<','0').replace('-','.').replace('>=','99'))):
    b = bins[k]
    print(f'  {k}: n={b["n"]} bias={b["b"]/b["n"]:+.3f} MAE={b["mae"]/b["n"]:.3f}')

# ===== 3. test_charts =====
print('\ntest_charts (生产路径):')
TC_DIR = os.path.join(_ROOT, 'data', 'test_charts')
for fn in sorted(os.listdir(TC_DIR)):
    if not fn.endswith('.json'): continue
    try:
        with open(os.path.join(TC_DIR, fn), 'rb') as f:
            chart_data, raw_text = appmod.load_chart_from_bytes(f.read())
        r, e = appmod.predict_one_chart(chart_data, level='IN', is_custom=True)
        r2, e2 = appmod.predict_one_chart(chart_data, level='AT', is_custom=True)
        if r and r2:
            print(f'  {fn[:40]}: IN={r["prediction"]:.2f} AT={r2["prediction"]:.2f}')
    except Exception as ex:
        print(f'  {fn[:40]}: ERR {ex}')
print('\nDONE')
