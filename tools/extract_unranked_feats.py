# -*- coding: utf-8 -*-
"""提取未上架高难谱特征 + 保存缓存 (下载完成后使用)
输出: data/phira/_feats_cache_unranked.npz (feats/labels/levels/names/ids)
"""
import os, sys, io, json
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator'
sys.path.insert(0, ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

CACHE = os.path.join(ROOT, 'data', 'phira', '_feats_cache_unranked.npz')
LIST = os.path.join(ROOT, 'data', 'phira', 'unranked_final_download.json')
JSON_DIR = os.path.join(ROOT, 'data', 'phira', 'json_unranked')

sel = json.load(open(LIST, encoding='utf-8'))

def parse_level(lv_str):
    s = str(lv_str).strip().upper().replace(' ', '')
    for lv in ['AT', 'IN', 'HD', 'EZ']:
        if s.startswith(lv):
            return lv
    return None

feats_list, labels, levels, names, ids = [], [], [], [], []
fails = []
for c in sel:
    cid = c['id']
    path = os.path.join(JSON_DIR, f'{cid}.json')
    if not os.path.exists(path):
        fails.append((cid, 'missing'))
        continue
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        cd, _ = load_chart_from_bytes(raw)
        if cd is None:
            fails.append((cid, 'parse None'))
            continue
        feats = extract_features(cd, speed=1.0)
        if not feats:
            fails.append((cid, 'no feats'))
            continue
        lv = parse_level(c.get('level', ''))
        if lv is None:
            lv = 'IN'
        feats_list.append(feats)
        labels.append(c.get('difficulty', 0))
        levels.append(lv)
        names.append(c.get('name', ''))
        ids.append(cid)
    except Exception as e:
        fails.append((cid, str(e)[:50]))

np.savez(CACHE, feats=np.array(feats_list, dtype=object),
         labels=np.array(labels), levels=np.array(levels, dtype=object),
         names=np.array(names, dtype=object), ids=np.array(ids))
print(f'提取成功 {len(feats_list)} / 失败 {len(fails)}')
for cid, err in fails[:15]:
    print(f'  FAIL {cid}: {err}')
print(f'缓存已保存: {CACHE}')
