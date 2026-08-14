import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import json
from collections import Counter
import numpy as np

path = os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json')
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

lines = data.get('judgeLineList', [])

# Check: how many type=4 notes are truly playable vs decorative?
# Look at positionX distribution of type=4
type4_positions = []
type4_lines = Counter()
type4_above = 0
type4_visible = []
type4_samples = []

for li, line in enumerate(lines):
    notes = line.get('notes', [])
    for n in notes:
        if n.get('type') == 4:
            type4_positions.append(n.get('positionX', 0))
            type4_lines[li] += 1
            type4_above += n.get('above', 1)
            type4_visible.append(n.get('visibleTime', 0))
            if len(type4_samples) < 5:
                type4_samples.append(n)
            # Check isFake
            if n.get('isFake', 0):
                print(f'  发现isFake=1 type=4 note!')

print(f'type=4 notes 总数: {len(type4_positions)}')
print(f'type=4 visibleTime 分布: min={min(type4_visible)}, max={max(type4_visible)}')
unique_visible = set(type4_visible)
print(f'type=4 visibleTime 唯一值: {unique_visible}')

pos_array = np.array(type4_positions)
print(f'type=4 positionX: min={pos_array.min():.1f}, max={pos_array.max():.1f}, mean={pos_array.mean():.1f}')
unique_positions = sorted(set(round(p, 1) for p in type4_positions))
print(f'type=4 唯一位置数: {len(unique_positions)}')
print(f'type=4 位置分布: {unique_positions[:10]}...{unique_positions[-5:]}')

# Distribution across lines
lines_with_type4 = len(type4_lines)
print(f'type=4 分布的线数: {lines_with_type4}/{len(lines)}')
print(f'type=4 per line: mean={np.mean(list(type4_lines.values())):.1f}, max={max(type4_lines.values())}')

# Compare: type=1 positions
type1_positions = []
for line in lines:
    for n in line.get('notes', []):
        if n.get('type') == 1:
            type1_positions.append(n.get('positionX', 0))
tp = np.array(type1_positions)
print(f'\ntype=1 (Tap) positionX: min={tp.min():.1f}, max={tp.max():.1f}, mean={tp.mean():.1f}')
print(f'type=1 唯一位置数: {len(set(round(p,1) for p in type1_positions))}')

# Check if type=4 notes might be something else - do they overlap with other notes in time?
# Check consecutive type=4 notes (quick succession might indicate they're not all playable)
type4_times = []
for line in lines:
    for n in line.get('notes', []):
        if n.get('type') == 4:
            st = n.get('startTime', [0,0,1])
            type4_times.append(st[0] * 4 + st[1] + (st[2]-1)/192.0)
type4_times.sort()
if len(type4_times) > 1:
    gaps = np.diff(type4_times)
    print(f'\ntype=4 最小间隔: {min(gaps):.4f} beats')
    print(f'type=4 间隔<0.01beats: {sum(1 for g in gaps if g < 0.01)}个')
    print(f'type=4 间隔<0.001beats: {sum(1 for g in gaps if g < 0.001)}个')

# Also check type=3 - could they be the holds?
type3_endtimes = []
type3_starttimes = []
for line in lines:
    for n in line.get('notes', []):
        if n.get('type') == 3:
            type3_starttimes.append(n.get('startTime', [0,0,1]))
            type3_endtimes.append(n.get('endTime', [0,0,1]))

if type3_starttimes:
    start_diffs = []
    for s, e in zip(type3_starttimes, type3_endtimes):
        s_b = s[0]*4 + s[1] + (s[2]-1)/192.0
        e_b = e[0]*4 + e[1] + (e[2]-1)/192.0
        start_diffs.append(e_b - s_b)
    print(f'\ntype=3 (假设Hold) 实际持续时间:')
    print(f'  非零时长: {sum(1 for d in start_diffs if d > 0)}/{len(start_diffs)}')

# Let's also check if there's a multiLineString or multiScale that relates to type=4
print(f'\n顶层key: multiline? {data.get("multiLineString", "N/A")[:50] if data.get("multiLineString") else "N/A"}')

# Check if type 4 is related to multiScale
ms = data.get('multiScale', None)
if ms:
    print(f'multiScale: {json.dumps(ms, ensure_ascii=False)[:200]}')
