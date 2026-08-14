# -*- coding: utf-8 -*-
"""统计 Retribution 音符级字段 (speed/visibleTime/isFake/alpha/type/yOffset) 分布"""
import json, collections

p = r'C:\Users\NaNK\Downloads\51030697.json'
with open(p, 'r', encoding='utf-8', errors='ignore') as f:
    data = json.load(f)

all_notes = []
for jl in data['judgeLineList']:
    all_notes.extend(jl.get('notes', []))
print(f'总音符: {len(all_notes)}')

for field in ['speed', 'type', 'isFake', 'alpha', 'visibleTime', 'above', 'yOffset', 'size']:
    cnt = collections.Counter()
    for n in all_notes:
        v = n.get(field)
        if isinstance(v, float):
            v = round(v, 3)
        cnt[v] += 1
    print(f'\n{field} 分布 (top10):')
    for v, c in cnt.most_common(10):
        print(f'  {v}: {c}')

# speed != 1 的音符数量
sp = [n.get('speed', 1) for n in all_notes]
n_non1 = sum(1 for s in sp if abs(s - 1) > 1e-6)
print(f'\nspeed != 1 的音符: {n_non1} / {len(all_notes)}')
import numpy as np
sp = np.array(sp)
print(f'speed: min={sp.min()} max={sp.max()} mean={sp.mean():.3f} std={sp.std():.3f}')
print('speed 直方图:', {round(k, 2): int(c) for k, c in collections.Counter(np.round(sp, 2)).most_common(15)})

# 长条 (type 3?) 的 visibleTime 与 endTime 差 (闪现程度)
holds = [n for n in all_notes if n.get('type') == 3]
print(f'\n长条数: {len(holds)}')
if holds:
    vt = [n.get('visibleTime', 999999) for n in holds]
    print(f'  长条 visibleTime: min={min(vt)} max={max(vt)}  <100的: {sum(1 for v in vt if v < 100)}')
    # visibleTime < 一定值的 (闪现长条)
    print(f'  visibleTime < 480 (闪现): {sum(1 for v in vt if v < 480)}')
