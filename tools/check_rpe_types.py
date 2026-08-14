import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import json
from collections import Counter

path = os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json')
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

lines = data.get('judgeLineList', [])

# 统计所有note的type值和对应数量
type_counter = Counter()
above_counter = Counter()
below_counter = Counter()
type_with_hold = Counter()

for line in lines:
    notes = line.get('notes', [])
    for n in notes:
        t = n.get('type', 0)
        type_counter[t] += 1
        if n.get('above', 1):
            above_counter[t] += 1
        else:
            below_counter[t] += 1
        if 'endTime' in n and n['endTime'] != n.get('startTime'):
            type_with_hold[t] += 1

print('所有type及其数量:')
for t, c in sorted(type_counter.items()):
    print(f'  type={t}: {c}个 (above={above_counter.get(t,0)}, below={below_counter.get(t,0)})')

# 看看type=4（我当初当成flick）的note长什么样
for line in lines:
    notes = line.get('notes', [])
    for n in notes:
        if n.get('type') == 4:
            print(f'\n一个type=4的完整note:')
            print(json.dumps(n, indent=2, ensure_ascii=False))
            break
    else:
        continue
    break

# 看看type=2（drag）的note
for line in lines:
    notes = line.get('notes', [])
    for n in notes:
        if n.get('type') == 2:
            print(f'\n一个type=2的完整note:')
            print(json.dumps(n, indent=2, ensure_ascii=False))
            break
    else:
        continue
    break

# 看看type=1的note
for line in lines:
    notes = line.get('notes', [])
    for n in notes:
        if n.get('type') == 1:
            print(f'\n一个type=1的完整note:')
            print(json.dumps(n, indent=2, ensure_ascii=False))
            break
    else:
        continue
    break

# 看看type=3的note
for line in lines:
    notes = line.get('notes', [])
    for n in notes:
        if n.get('type') == 3:
            print(f'\n一个type=3的完整note:')
            print(json.dumps(n, indent=2, ensure_ascii=False))
            break
    else:
        continue
    break

# 另外看一下标准谱面的type映射
print('\n\n=== 对比：加载一个标准谱面看看type定义 ===')
std_path = os.path.join(_ROOT, 'data', 'chart', 'Rrharil.TeamGrimoire.0', 'AT.json')
with open(std_path, 'r', encoding='utf-8') as f:
    std_data = json.load(f)

std_type_counter = Counter()
for line in std_data.get('judgeLineList', []):
    for n in line.get('notesAbove', []) + line.get('notesBelow', []):
        std_type_counter[n.get('type', 0)] += 1

print('标准谱面的type分布:')
for t, c in sorted(std_type_counter.items()):
    print(f'  type={t}: {c}个')
