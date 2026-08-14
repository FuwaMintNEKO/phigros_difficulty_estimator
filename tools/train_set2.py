# -*- coding: utf-8 -*-
"""分层采样: 按定数段分层, 段内按count+rating排序, 总1000"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
meta = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'unranked_all.json'), encoding='utf-8'))
import glob
files = set(int(os.path.basename(f)[:-5]) for f in glob.glob(os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '*.json')))
SPECIAL_LV = ['SP', 'AB', 'END', 'LOVE', 'HL', 'JLL', 'Legency', 'Lv.避', 'Lv.小奏', 'Lv.?']
def is_special_lv(s):
    s = (s or '').upper()
    return any(k.upper() in s for k in SPECIAL_LV)
cands = []
for c in meta:
    cid = c['id']
    if cid not in files or c.get('ranked'): continue
    if is_special_lv(c.get('level')): continue
    d = c.get('difficulty')
    if d is None: continue
    try: d = float(d)
    except: continue
    if not (5 <= d <= 20): continue
    rc = c.get('ratingCount', 0); rt = c.get('rating', 0)
    if rc <= 0 or rt is None: continue
    cands.append({'id': cid, 'name': c.get('name',''), 'level': c.get('level',''), 'diff': d,
                  'rating': rt, 'count': rc})
# 分层: 段分配
SEGS = [(5,13,'5-13'),(13,14,'13-14'),(14,15,'14-15'),(15,16,'15-16'),(16,16.5,'16-16.5'),(16.5,17,'16.5-17'),(17,17.7,'17-17.7'),(17.7,99,'17.7+')]
# 先看各段可用数
for lo, hi, tag in SEGS:
    n = sum(1 for c in cands if lo <= c['diff'] < hi)
    print(f'段{tag:<8} 可用: {n}')
# 按官谱982的段分布比例分配? 官谱段分布:
# 用简单配额: 高段优先多给
QUOTA = [(5,13,120),(13,14,120),(14,15,200),(15,16,240),(16,16.5,120),(16.5,17,100),(17,17.7,60),(17.7,99,40)]
# 调整到1000
total_q = sum(q for _,_,q in QUOTA)
print('\n配额合计:', total_q)
# 实际分配 (按可用数cap)
alloc = []
for (lo, hi, tag), (_,_,q) in zip(SEGS, QUOTA):
    seg = [c for c in cands if lo <= c['diff'] < hi]
    seg.sort(key=lambda c: (-c['count'], -c['rating']))
    take = min(q, len(seg))
    alloc.extend(seg[:take])
    print(f'段{tag:<8} 取{take}')
print('\n总取样:', len(alloc))
d = np.array([c['diff'] for c in alloc])
print(f'diff: P50={np.percentile(d,50):.1f} P90={np.percentile(d,90):.1f} 16.5+={np.sum(d>=16.5)} 17+={np.sum(d>=17)} 17.7+={np.sum(d>=17.7)}')
print(f'count范围: min={min(c["count"] for c in alloc)} max={max(c["count"] for c in alloc)}')
print(f'rating范围: min={min(c["rating"] for c in alloc):.3f}')
# 输出到csv
import csv as _csv
out = os.path.join(_ROOT, 'data', 'phira', 'train_unranked_1000.csv')
with open(out, 'w', encoding='utf-8-sig', newline='') as f:
    w = _csv.writer(f)
    w.writerow(['id','name','level','diff','rating','ratingCount'])
    for c in alloc:
        w.writerow([c['id'], c['name'], c['level'], c['diff'], round(c['rating'],4), c['count']])
print(f'已保存: {out}')
print('DONE')