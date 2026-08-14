# -*- coding: utf-8 -*-
"""检查完整版 Retribution 音符的 speed / visibleTime / holdTime 字段分布"""
import json, os
from collections import Counter

DL = 'C:/Users/NaNK/Downloads'
out = os.path.join(DL, 'Retribution_FULL.json')
with open(out, 'rb') as f:
    data = json.load(f)

jls = data.get('judgeLineList', [])
all_notes = []
for line in jls:
    all_notes.extend(line.get('notesAbove', []))
    all_notes.extend(line.get('notesBelow', []))

print('总音符:', len(all_notes), ' 线数:', len(jls))

# speed 字段分布
speeds = Counter()
has_speed_key = 0
visible_times = Counter()
has_vt_key = 0
holds = Counter()
for n in all_notes:
    if 'speed' in n:
        has_speed_key += 1
        speeds[n['speed']] += 1
    if 'visibleTime' in n:
        has_vt_key += 1
        vt = n['visibleTime']
        if vt >= 999999:
            visible_times['default(999999+)'] += 1
        else:
            visible_times[round(vt, 2)] += 1
    holds[n.get('type')] += 1

print('有 speed 字段的音符:', has_speed_key)
print('speed 分布(前10):', speeds.most_common(10))
print('有 visibleTime 字段的音符:', has_vt_key)
print('visibleTime 分布(非默认前10):', [(k, v) for k, v in visible_times.most_common(12)])
print('type 分布:', dict(holds))

# 检查 hold 音符的 visibleTime 与 speed
hold_notes = [n for n in all_notes if n.get('type') == 3]
print('hold 音符数:', len(hold_notes))
if hold_notes:
    hv = Counter()
    hs = Counter()
    for n in hold_notes:
        vt = n.get('visibleTime', 'no_key')
        hv[vt] += 1
        hs[n.get('speed', 'no_key')] += 1
    print('hold visibleTime 分布(前8):', hv.most_common(8))
    print('hold speed 分布(前8):', hs.most_common(8))
