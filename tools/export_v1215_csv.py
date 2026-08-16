# -*- coding: utf-8 -*-
"""v13.0 全量预测导出 (生产模型 6dim_model_v13.pkl, 原始GB+Boost无规则)
ranked:  feats_cache_v11.pkl (上架+特殊 615 首, diff>10)  -> v130_ranked_predictions.csv
unranked: json_unranked 957 首 (unranked_final_download.json 元数据) -> v130_unranked_predictions.csv
统一调用 app.predict_from_feats / app.predict_one_chart, 与网页口径一致。
"""
import os, sys, json, csv, io, pickle, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)

from unified_parser import load_chart_from_bytes
import app as app_mod

DATA = os.path.join(_ROOT, 'data', 'phira')


def level_key(s):
    s = (s or '').upper()
    if 'AT' in s: return 'AT'
    if 'IN' in s: return 'IN'
    if 'HD' in s: return 'HD'
    return 'IN'


# ---------- ranked ----------
def export_ranked():
    with open(os.path.join(DATA, 'feats_cache_v11.pkl'), 'rb') as f:
        cache = pickle.load(f)
    ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
    out = os.path.join(DATA, 'v130d_ranked_predictions.csv')
    rows = []
    for r in ranked:
        feats = dict(r['feats'])
        lv = level_key(r['level'])
        if lv == 'IN':
            for k, d in app_mod.DOMAIN_DELTA.items():
                if k in feats: feats[k] = feats[k] - d
        pred = float(app_mod.predict_from_feats(feats, lv, is_custom=True)[0])
        d = round(r['diff'], 1)
        ts = '+'.join(app_mod.compute_tags(feats)) if app_mod.compute_tags(feats) else '-'
        kt = app_mod.kyou_type_for(feats, r['name'], True)
        hr = r['feats'].get('hold_count', 0) / max(r['feats'].get('total_notes', 1), 1)
        rows.append([r['id'], r['name'], r['level'], d, round(pred, 2), round(pred - d, 2),
                     r['feats'].get('multi_finger_3plus_events', 0), round(hr, 2), ts,
                     (kt or {}).get('type', '')])
    with open(out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'name', 'level', 'diff', 'pred', 'err', 'mf3', 'hold%', 'tags', 'kyou_type'])
        w.writerows(rows)
    print('[ranked] %d 行 -> %s' % (len(rows), out), flush=True)


# ---------- unranked ----------
def export_unranked():
    meta = {c['id']: c for c in json.load(open(os.path.join(DATA, 'unranked_final_download.json'), encoding='utf-8'))}
    jdir = os.path.join(DATA, 'json_unranked')
    out = os.path.join(DATA, 'v130d_unranked_predictions.csv')
    rows, fails = [], []
    t0 = time.time()
    for i, fn in enumerate(sorted(os.listdir(jdir))):
        if not fn.endswith('.json'):
            continue
        cid = int(fn[:-5])
        info = meta.get(cid, {})
        try:
            with open(os.path.join(jdir, fn), 'rb') as f:
                raw = f.read()
            cd, pe = load_chart_from_bytes(raw)
            if cd is None:
                fails.append((cid, 'parse None')); continue
            lv = 'AT' if info.get('difficulty', 0) >= 16.5 else 'IN'
            is_custom = app_mod.is_custom_chart(cd, pe)
            r, e = app_mod.predict_one_chart(cd, speed=1.0, level=lv, is_custom=is_custom)
            if r is None:
                fails.append((cid, e)); continue
            rows.append({
                'id': cid, 'name': info.get('name', ''),
                'diff': info.get('difficulty', 0), 'level': info.get('level', ''),
                'pred': round(r['prediction'], 2), 'gb': round(r['gb'], 2),
                'boost': round(r['boost'], 2),
                'notes': r.get('total_notes'), 'dur': r.get('duration_sec'),
            })
        except Exception as ex:
            fails.append((cid, str(ex)[:80]))
        if (i + 1) % 200 == 0:
            print('[unranked] %d/957 done, %.0fs' % (i + 1, time.time() - t0), flush=True)
    with open(out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'name', 'diff', 'level', 'pred', 'gb', 'boost', 'notes', 'dur'], extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print('[unranked] 成功 %d, 失败 %d, %.0fs -> %s' % (len(rows), len(fails), time.time() - t0, out), flush=True)
    for cid, e in fails[:15]:
        print('  FAIL %s: %s' % (cid, e), flush=True)


if __name__ == '__main__':
    export_ranked()
    export_unranked()
    print('ALL DONE', flush=True)
