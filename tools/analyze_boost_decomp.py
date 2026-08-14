# -*- coding: utf-8 -*-
"""boost 贡献分解: 定位 14-16 区间高估的推手特征

对上架谱按组 (14-16高估 / 16-17 / >=17 / 全14-16) 平均各 boost 特征贡献,
对比找出"低区间高、高区间不高"的特征 → 降权候选
"""
import os, sys, json, csv, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
import app

# 只重算一次特征: 读上架谱 json
charts = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
ranked_ids = {c['id'] for c in charts['上架']}
ranked_info = {c['id']: c for c in charts['上架']}

rows = []
for r in csv.DictReader(open(os.path.join(_ROOT, 'data', 'phira', 'ranked_compare.csv'), encoding='utf-8-sig')):
    rows.append(r)

# 每个谱: 特征 + 各 boost 贡献
def boost_contribs(feats):
    out = {}
    total = 0.0
    cap_default = app.CAPS.get('_default', None)
    for fname, bl, co in app.MANUAL_FLAT:
        v = feats.get(fname, 0)
        pv = app.P95.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t:
            continue
        e = v / t - 1.0
        c = app.CAPS.get(fname, cap_default)
        if c is not None and e > c:
            e = c
        x = co * (e ** 0.70)
        if v > max(app.P99.get(fname, 0), bl * 0.5):
            pe = v / max(app.P99.get(fname, 0), bl * 0.5) - 1.0
            if c is not None and pe > c:
                pe = c
            x += co * max(0, pe) ** 0.70 * 0.5
        total += x
        out[fname] = x
    return out, total

records = []  # (name, diff, pred, bias, boost, contribs)
for r in rows:
    cid = int(r['id'])
    path = os.path.join(_ROOT, 'data', 'phira', 'json', f'{cid}.json')
    if not os.path.exists(path):
        continue
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        cd, pe = load_chart_from_bytes(raw)
        if cd is None:
            continue
        feats = extract_features(cd)
        if not feats:
            continue
        contribs, total = boost_contribs(feats)
        records.append({
            'name': r['name'], 'diff': float(r['diff']), 'pred': float(r['pred']),
            'bias': float(r['pred']) - float(r['diff']), 'boost': total, 'contribs': contribs,
        })
    except Exception:
        continue
print(f'重算特征谱数: {len(records)}')

# 分组
g1 = [x for x in records if 14 <= x['diff'] < 16 and x['bias'] > 0.3]   # 14-16 高估组
g2 = [x for x in records if 14 <= x['diff'] < 16]                        # 14-16 全体
g3 = [x for x in records if 16 <= x['diff'] < 17]                        # 16-17
g4 = [x for x in records if x['diff'] >= 17]                             # >=17

def avg(grp, fname):
    vals = [x['contribs'].get(fname, 0) for x in grp]
    return np.mean(vals)

fnames = [f[0] for f in app.MANUAL_FLAT]
print(f'\n{"特征":<28} {"14-16高估":>9} {"14-16全":>8} {"16-17":>8} {">=17":>8}  判断')
print('-' * 80)
cands = []
for fn in fnames:
    a1, a2, a3, a4 = avg(g1, fn), avg(g2, fn), avg(g3, fn), avg(g4, fn)
    if a1 < 0.05 and a4 < 0.05:
        continue
    # 候选: 14-16 高估组贡献大, 且明显高于 16-17 / >=17
    if a1 > 0.15 and a1 > a3 * 1.5 and a1 > a4 * 1.5:
        tag = '★ 候选降权'
        cands.append((fn, a1, a3, a4))
    else:
        tag = ''
    print(f'{fn:<28} {a1:>9.3f} {a2:>8.3f} {a3:>8.3f} {a4:>8.3f}  {tag}')

print('\nboost 总量对比: 14-16高估=%.3f, 14-16全=%.3f, 16-17=%.3f, >=17=%.3f' % (
    np.mean([x['boost'] for x in g1]), np.mean([x['boost'] for x in g2]),
    np.mean([x['boost'] for x in g3]), np.mean([x['boost'] for x in g4])))
