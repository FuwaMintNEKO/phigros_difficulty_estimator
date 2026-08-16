# -*- coding: utf-8 -*-
"""v13.0 未上架全量预测: json_unranked_4star 5901 首 -> v130_unranked_4star_predictions.csv
元数据: unranked_4star_list_v12.csv 为主, 缺失 id 用 unranked_all.json 补
预测口径: app.predict_one_chart (与网页一致), level=AT if diff>=16.5 else IN
"""
import os, sys, json, csv, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)

from unified_parser import load_chart_from_bytes
import app as app_mod

DATA = os.path.join(_ROOT, 'data', 'phira')
JDIR = os.path.join(DATA, 'json_unranked_4star')
OUT = os.path.join(DATA, 'v130_unranked_4star_predictions.csv')

# 元数据: 列表为主, all.json 兜底
meta = {}
for r in csv.reader(open(os.path.join(DATA, 'unranked_4star_list_v12.csv'), encoding='utf-8-sig')):
    if not r or not r[0].strip().isdigit(): continue
    meta[int(r[0])] = {'name': r[1], 'level': r[2], 'diff': float(r[3] or 0)}
for c in json.load(open(os.path.join(DATA, 'unranked_all.json'), encoding='utf-8')):
    cid = c.get('id')
    if cid is not None and cid not in meta:
        meta[cid] = {'name': c.get('name', ''), 'level': c.get('level', ''),
                     'diff': float(c.get('difficulty', 0) or 0)}

files = sorted([f for f in os.listdir(JDIR) if f.endswith('.json')],
               key=lambda x: int(x[:-5]))
rows, fails = [], []
t0 = time.time()
for i, fn in enumerate(files):
    cid = int(fn[:-5])
    info = meta.get(cid, {})
    try:
        with open(os.path.join(JDIR, fn), 'rb') as f:
            raw = f.read()
        cd, pe = load_chart_from_bytes(raw)
        if cd is None:
            fails.append((cid, 'parse None')); continue
        lv = 'AT' if info.get('diff', 0) >= 16.5 else 'IN'
        is_custom = app_mod.is_custom_chart(cd, pe)
        r, e = app_mod.predict_one_chart(cd, speed=1.0, level=lv, is_custom=is_custom)
        if r is None:
            fails.append((cid, e)); continue
        rows.append({
            'id': cid, 'name': info.get('name', ''),
            'diff': info.get('diff', 0), 'level': info.get('level', ''),
            'pred': round(r['prediction'], 2), 'gb': round(r['gb'], 2),
            'boost': round(r['boost'], 2),
            'notes': r.get('total_notes'), 'dur': r.get('duration_sec'),
        })
    except Exception as ex:
        fails.append((cid, str(ex)[:80]))
    if (i + 1) % 300 == 0:
        el = time.time() - t0
        print('[unranked-full] %d/%d done, %.0fs, 失败%d' % (i + 1, len(files), el, len(fails)), flush=True)

with open(OUT, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'name', 'diff', 'level', 'pred', 'gb', 'boost', 'notes', 'dur'], extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)
print('[unranked-full] 成功 %d, 失败 %d, %.0fs -> %s' % (len(rows), len(fails), time.time() - t0, OUT), flush=True)
for cid, e in fails[:20]:
    print('  FAIL %s: %s' % (cid, e), flush=True)
print('ALL DONE', flush=True)
