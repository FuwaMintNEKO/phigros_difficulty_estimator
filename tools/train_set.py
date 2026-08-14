# -*- coding: utf-8 -*-
"""构建训练集方案: 高游玩+高评分 unranked 谱筛选 (标签=社区定数difficulty)"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
meta = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'unranked_all.json'), encoding='utf-8'))

# 筛选条件: 有文件(json_unranked_4star), 非特殊谱面, diff合理
import glob
files = set(int(os.path.basename(f)[:-5]) for f in glob.glob(os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '*.json')))
print('已下载json文件数:', len(files))

# 特殊level标记 (SP/AB/END/LOVE/HL/JLL/Legency等非标准)
SPECIAL_LV = ['SP', 'AB', 'END', 'LOVE', 'HL', 'JLL', 'Legency', 'Lv.避', 'Lv.小奏', 'Lv.?']
def is_special_lv(s):
    s = (s or '').upper()
    return any(k.upper() in s for k in SPECIAL_LV)

cands = []
for c in meta:
    cid = c['id']
    if cid not in files: continue
    if c.get('ranked'): continue
    if is_special_lv(c.get('level')): continue
    d = c.get('difficulty')
    if d is None: continue
    try: d = float(d)
    except: continue
    if not (5 <= d <= 20): continue  # 合理定数范围
    rc = c.get('ratingCount', 0); rt = c.get('rating', 0)
    if rc <= 0 or rt is None: continue
    cands.append({'id': cid, 'name': c.get('name',''), 'level': c.get('level',''), 'diff': d,
                  'rating': rt, 'count': rc})
print(f'候选: {len(cands)}')
diffs = np.array([c['diff'] for c in cands])
print(f'diff: P25={np.percentile(diffs,25):.1f} P50={np.percentile(diffs,50):.1f} P75={np.percentile(diffs,75):.1f} P90={np.percentile(diffs,90):.1f} max={diffs.max():.1f}')
print(f'  16.5+: {np.sum(diffs>=16.5)}, 17+: {np.sum(diffs>=17)}, 17.7+: {np.sum(diffs>=17.7)}')
# 整数diff占比
int_ratio = np.mean(np.abs(diffs - np.round(diffs)) < 1e-6)
print(f'整数diff占比: {int_ratio*100:.0f}%')
# 按 count 排序取 top N
cands.sort(key=lambda c: (-c['count'], -c['rating']))
for N in [500, 800, 1000, 1200]:
    top = cands[:N]
    d = np.array([c['diff'] for c in top])
    print(f'\ntop{N} (按count): diff P50={np.percentile(d,50):.1f} P90={np.percentile(d,90):.1f} 16.5+={np.sum(d>=16.5)} 17+={np.sum(d>=17)} 17.7+={np.sum(d>=17.7)} count_min={top[-1]["count"]} rating_min={min(c["rating"] for c in top):.3f}')
print('DONE')