# -*- coding: utf-8 -*-
"""RPE多线谱: 判定线屏幕位置来源 (看RPE v3转换/多线机制)"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
raw = json.load(open(p, encoding='utf-8'))
# 全部线的 posControl (缩放) 和是否有全局位置
print('=== 前3条线的完整控制字段 ===')
for jl in raw['judgeLineList'][:3]:
    print(f'\n线 {jl.get("Name")}:')
    for k in ['posControl', 'yControl', 'Group', 'father', 'x', 'y', 'anchor']:
        if k in jl:
            print(f'  {k}: {json.dumps(jl[k], ensure_ascii=False)[:150]}')
# 线的数量与音符分布
from collections import Counter
notes_per_line = Counter()
line_has = {}
for li, jl in enumerate(raw['judgeLineList']):
    ns = jl.get('notes', [])
    notes_per_line[li] = len(ns)
    if ns:
        line_has[li] = np.mean([n.get('positionX', 0) for n in ns])
print(f'\n有音符的线: {len(line_has)}/{len(raw["judgeLineList"])}')
# 关键: RPE 多线谱中, 判定线之间的间距由什么决定?
# 看音符 positionX 的绝对值分布 (如果线都居中, positionX就是屏幕位置)
all_px = []
for jl in raw['judgeLineList']:
    for n in jl.get('notes', []):
        all_px.append(n.get('positionX', 0))
all_px = np.array(all_px)
print(f'\n所有音符 positionX: P10={np.percentile(all_px,10):.1f} P50={np.percentile(all_px,50):.1f} P90={np.percentile(all_px,90):.1f} min={all_px.min():.1f} max={all_px.max():.1f}')
# 相邻音符(时间序)的 positionX 跳变
notes_t = []
for jl in raw['judgeLineList']:
    for n in jl.get('notes', []):
        st = n.get('startTime')
        if isinstance(st, list) and len(st) == 3:
            t = (st[0]*4.0 + st[1]*(4.0/st[2])) * 8.0
        else:
            t = 0
        notes_t.append((t, n.get('positionX', 0), n.get('type')))
notes_t.sort()
px = np.array([x[1] for x in notes_t])
dpx = np.abs(np.diff(px))
print(f'\n相邻音符 |ΔpositionX|: P50={np.percentile(dpx,50):.1f} P90={np.percentile(dpx,90):.1f} max={dpx.max():.1f}')
print(f'  Δ>5: {np.sum(dpx>5)}  Δ>8: {np.sum(dpx>8)}')
print('DONE')