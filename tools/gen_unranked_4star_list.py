# -*- coding: utf-8 -*-
"""只生成未上架4星清单 CSV (文件已复制完成)"""
import os, sys, json, io, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(_ROOT, 'data', 'phira', 'unranked_final_download.json')
meta = json.load(open(META, encoding='utf-8'))
sel = [c for c in meta if (c.get('rating') or 0) >= 0.8]

def read_pred(path):
    preds = {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f)
        head = next(rd)
        idx = {h: i for i, h in enumerate(head)}
        for c in rd:
            if len(c) < len(head): continue
            try: preds[int(c[idx['id']])] = {h: c[idx[h]] for h in head}
            except Exception: pass
    return preds
pred_map = read_pred(os.path.join(_ROOT, 'data', 'phira', 'v112_unranked_predictions_v2.csv'))

out_csv = os.path.join(_ROOT, 'data', 'phira', 'unranked_4star_list.csv')
with open(out_csv, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'name', 'level', 'difficulty', 'rating', 'ratingCount', 'pred', 'err', 'mf3', 'dens', 'nps', 'notes'])
    for c in sorted(sel, key=lambda x: -x['rating']):
        p = pred_map.get(c['id'], {})
        w.writerow([c['id'], c['name'], c['level'], c.get('difficulty'), round(c.get('rating', 0), 4), c.get('ratingCount', 0),
                    p.get('pred', ''), p.get('err', ''), p.get('mf3', ''), p.get('dens', ''), p.get('nps', ''), p.get('notes', '')])
print(f'清单已写入: {out_csv} ({len(sel)} 行)')
