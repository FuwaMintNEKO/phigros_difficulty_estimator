# -*- coding: utf-8 -*-
"""预测未上架谱 957 张 (生产模型 3类) → data/phira/unranked_predictions.csv
与 analyze_phira.py 同路径, 但读 json_unranked 目录"""
import os, sys, json, csv, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from unified_parser import load_chart_from_bytes
import app

LIST = os.path.join(_ROOT, 'data', 'phira', 'unranked_final_download.json')
JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked')
OUT = os.path.join(_ROOT, 'data', 'phira', 'unranked_predictions.csv')


def main():
    meta = {c['id']: c for c in json.load(open(LIST, encoding='utf-8'))}
    rows, fails = [], []
    for fn in sorted(os.listdir(JSON_DIR)):
        if not fn.endswith('.json'):
            continue
        cid = int(fn[:-5])
        info = meta.get(cid, {})
        path = os.path.join(JSON_DIR, fn)
        try:
            with open(path, 'rb') as f:
                raw = f.read()
            cd, pe = load_chart_from_bytes(raw)
            if cd is None:
                fails.append((cid, 'parse None'))
                continue
            lv = 'AT' if info.get('difficulty', 0) >= 16.5 else 'IN'
            is_custom = app.is_custom_chart(cd, pe)
            r, e = app.predict_one_chart(cd, speed=1.0, level=lv, is_custom=is_custom)
            if r is None:
                fails.append((cid, e))
                continue
            rows.append({
                'id': cid, 'name': info.get('name', ''),
                'diff': info.get('difficulty', 0), 'level': info.get('level', ''),
                'pred': r['prediction'], 'gb': r['gb'], 'boost': r['boost'],
                'notes': r.get('total_notes'), 'dur': r.get('duration_sec'),
            })
        except Exception as ex:
            fails.append((cid, str(ex)[:80]))

    with open(OUT, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [], extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'未上架预测成功 {len(rows)}, 失败 {len(fails)}')
    for cid, e in fails[:10]:
        print(f'  FAIL {cid}: {e}')
    print(f'已保存: {OUT}')


if __name__ == '__main__':
    main()
