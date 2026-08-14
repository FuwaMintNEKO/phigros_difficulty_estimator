# -*- coding: utf-8 -*-
"""v11 特征缓存: 官谱982 + 上架615 一次性提取, 存 pickle 供所有实验秒级复用
输出: data/phira/feats_cache_v11.pkl
结构: {'official': [{'name','level','diff','feats'},...], 'ranked': [{'id','name','level','diff','feats'},...]}
"""
import os, sys, json, pickle, csv as _csv
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from unified_parser import load_chart_from_bytes

# ===== 官谱 =====
song_difficulties = load_difficulty_tsv(os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv'))
chart_files = find_chart_files(os.path.join(_ROOT, 'data', 'chart'))
official = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    for lv in ['EZ','HD','IN','AT']:
        if lv in info['levels'] and lv in diffs:
            try:
                cd = load_chart_json(info['levels'][lv])
                feats = extract_features(cd)
                if feats:
                    official.append({'name': fn, 'level': lv, 'diff': diffs[lv], 'feats': feats})
            except Exception:
                pass
print(f'官谱特征: {len(official)}')

# ===== 上架谱 =====
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
ranked = []
for fn in sorted(os.listdir(JSON_DIR)):
    if not fn.endswith('.json'): continue
    cid = int(fn[:-5])
    meta = meta_by_id.get(cid, {})
    try:
        with open(os.path.join(JSON_DIR, fn), 'rb') as f:
            chart_data, raw_text = load_chart_from_bytes(f.read())
        if chart_data is None: continue
        feats = extract_features(chart_data, speed=1.0)
        if not feats: continue
        old = pred_old.get(cid, {})
        diff = None
        try: diff = float(old.get('diff')) if old.get('diff') else None
        except Exception: diff = None
        ranked.append({'id': cid, 'name': meta.get('name',''), 'level': meta.get('level','IN'),
                       'diff': diff, 'feats': feats})
    except Exception:
        pass
print(f'上架谱特征: {len(ranked)}')

out = os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl')
with open(out, 'wb') as f:
    pickle.dump({'official': official, 'ranked': ranked}, f)
print(f'已保存: {out}')
