# -*- coding: utf-8 -*-
"""官谱特征邻居法: 用特征最相似的官谱定数估计自制谱"实际定数"

- 官谱: data/chart + difficulty.tsv (特征提取, 与训练一致)
- 自制谱: data/phira/json/{id}.json
- 对每张自制谱, 在 GB 特征空间找同一 level 的 K 个最近官谱,
  以距离加权估计实际定数 → 与社区定数对比 (社区普遍高估假设的检验)

用法: python tools/neighbor_official_estimate.py [--min-diff 14]
输出: data/phira/neighbor_estimate.csv
"""
import os, sys, json, csv, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from unified_parser import load_chart_from_bytes
import app

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json')
LV_ORDER = ['EZ', 'HD', 'IN', 'AT']
K = 8

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--min-diff', type=float, default=0.0)
args = ap.parse_args()

# ===== 官谱特征 =====
charts = find_chart_files(CHART_DIR)
diffs = load_difficulty_tsv(TSV)
official = []
for fn, info in charts.items():
    for lv in LV_ORDER:
        if lv not in info['levels']:
            continue
        d = (diffs.get(info['song_id']) or {}).get(lv)
        if d is None:
            continue
        try:
            feats = extract_features(load_chart_json(info['levels'][lv]))
            if feats:
                official.append({'folder': fn, 'level': lv, 'diff': d, 'feats': feats})
        except Exception:
            pass
print(f'官谱样本: {len(official)}')

# 特征名对齐 (app 里保存的 GB 特征名)
gb_names = app.FN
print(f'GB特征数: {len(gb_names)}')

def vec(feats):
    return np.array([feats.get(nn, 0) for nn in gb_names])

# 官谱按 level 分组, 标准化
by_lv = {lv: [] for lv in LV_ORDER}
for o in official:
    by_lv[o['level']].append(o)
scalers = {}
off_vec = {}
for lv, lst in by_lv.items():
    if not lst:
        continue
    X = np.array([vec(o['feats']) for o in lst])
    sc = StandardScaler().fit(X)
    scalers[lv] = sc
    off_vec[lv] = sc.transform(X)

# ===== 自制谱预测 + 邻居估计 =====
meta = {}
charts_meta = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
for lst in charts_meta.values():
    for c in lst:
        meta[c['id']] = c

rows = []
for fn in sorted(os.listdir(JSON_DIR)):
    if not fn.endswith('.json'):
        continue
    cid = int(fn[:-5])
    info = meta.get(cid, {})
    diff = info.get('difficulty', 0)
    if diff < args.min_diff:
        continue
    try:
        with open(os.path.join(JSON_DIR, fn), 'rb') as f:
            raw = f.read()
        cd, pe = load_chart_from_bytes(raw)
        if cd is None:
            continue
        feats = extract_features(cd)
        if not feats:
            continue
        lv = 'AT' if diff >= 16.5 else 'IN'
        r, e = app.predict_one_chart(cd, speed=1.0, level=lv)
        if r is None:
            continue
        x = vec(feats)
        if lv not in scalers or not len(off_vec[lv]):
            continue
        xs = scalers[lv].transform(x.reshape(1, -1))[0]
        D = off_vec[lv] - xs
        dists = np.sqrt((D ** 2).sum(axis=1))
        nn_idx = np.argsort(dists)[:K]
        nn = [by_lv[lv][i] for i in nn_idx]
        w = 1.0 / (dists[nn_idx] + 1e-6)
        neigh_est = float(np.average([n['diff'] for n in nn], weights=w))
        rows.append({
            'id': cid,
            'name': (cd.get('META') or {}).get('name') or info.get('name', ''),
            'diff': diff, 'level': info.get('level', ''),
            'pred': r['prediction'],
            'neigh_est': neigh_est,
            'neigh': ' / '.join(f"{n['folder']}({n['diff']:.1f})" for n in nn[:4]),
        })
    except Exception:
        pass

# 输出
out_csv = os.path.join(_ROOT, 'data', 'phira', 'neighbor_estimate.csv')
with open(out_csv, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'name', 'diff', 'level', 'pred', 'neigh_est', 'neigh'])
    w.writeheader()
    w.writerows(rows)

print(f'自制谱样本: {len(rows)}')
print(f'{"谱面名":<28} {"社区":>5} {"预测":>6} {"邻居估计":>8} {"预测-社区":>9} {"邻居-社区":>9}')
print('-' * 90)
import statistics
b1, b2 = [], []
for r in sorted(rows, key=lambda x: -x['diff']):
    b1.append(r['pred'] - r['diff'])
    b2.append(r['neigh_est'] - r['diff'])
    print(f'{str(r["name"])[:28]:<28} {r["diff"]:>5.1f} {r["pred"]:>6.2f} {r["neigh_est"]:>8.2f} '
          f'{r["pred"]-r["diff"]:>+9.2f} {r["neigh_est"]-r["diff"]:>+9.2f}')
if b1:
    print(f'\n模型: 均值偏差={statistics.mean(b1):+.3f}  MAE={statistics.mean(abs(b) for b in b1):.3f}')
    print(f'邻居: 均值偏差={statistics.mean(b2):+.3f}  MAE={statistics.mean(abs(b) for b in b2):.3f}')
    print(f'已保存 {out_csv}')
