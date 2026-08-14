import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import json

path = os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json')
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

lines = data.get('judgeLineList', [])

# 看一条type=2的note的完整结构
# type=2有startTime≠endTime，意味着它有路径/轨迹
type2_sample = None
type3_sample = None
type4_sample = None
type1_sample = None

for line in lines:
    for n in line.get('notes', []):
        t = n.get('type')
        if t == 2 and type2_sample is None:
            type2_sample = n
        if t == 3 and type3_sample is None:
            type3_sample = n
        if t == 4 and type4_sample is None:
            type4_sample = n
        if t == 1 and type1_sample is None:
            type1_sample = n
    if all([type1_sample, type2_sample, type3_sample, type4_sample]):
        break

print('=== 各类型note对比 ===')
for label, n in [('type=1 Tap', type1_sample), ('type=2', type2_sample), ('type=3', type3_sample), ('type=4 (你说是Drag)', type4_sample)]:
    print(f'\n{label}:')
    print(f'  startTime: {n.get("startTime")}')
    print(f'  endTime: {n.get("endTime")}')
    same = n['startTime'] == n['endTime']
    print(f'  start==end? {same}')
    if not same:
        start = n['startTime'][0]*4 + n['startTime'][1] + (n['startTime'][2]-1)/192
        end = n['endTime'][0]*4 + n['endTime'][1] + (n['endTime'][2]-1)/192
        print(f'  持续时间: {end-start:.4f} beats')
    print(f'  positionX: {n.get("positionX")}')
    print(f'  above: {n.get("above")}')
    print(f'  isFake: {n.get("isFake")}')
    print(f'  visibleTime: {n.get("visibleTime")}')

# 统计有endTime != startTime的note数量（有轨迹/持续时间的）
print('\n\n=== 有持续时间的note ===')
for t in [1,2,3,4]:
    count = 0
    for line in lines:
        for n in line.get('notes', []):
            if n.get('type') == t:
                if n['startTime'] != n['endTime']:
                    count += 1
    print(f'  type={t}: {count}个 startTime≠endTime')

# positionX范围对比
print('\n\n=== 各类型positionX范围 ===')
for t in [1,2,3,4]:
    positions = []
    for line in lines:
        for n in line.get('notes', []):
            if n.get('type') == t:
                positions.append(n.get('positionX', 0))
    if positions:
        print(f'  type={t}: range=[{min(positions):.1f}, {max(positions):.1f}], 唯一位置={len(set(round(p,1) for p in positions))}')
