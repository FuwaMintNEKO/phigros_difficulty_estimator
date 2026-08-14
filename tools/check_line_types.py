import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import json

path = os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json')
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

lines = data.get('judgeLineList', [])

# 检查每条线的notes类型构成
for i, line in enumerate(lines):
    notes = line.get('notes', [])
    types = {}
    for n in notes:
        t = n.get('type', 0)
        types[t] = types.get(t, 0) + 1
    if types:
        # 标记只有type=4的线
        has_only_4 = all(t == 4 for t in types)
        has_tap = any(t == 1 for t in types)
        has_drag = any(t == 2 for t in types)
        has_3 = any(t == 3 for t in types)
        has_4 = any(t == 4 for t in types)
        total = sum(types.values())
        
        label = ''
        if has_only_4: label = ' ⚠️ 只有type=4'
        elif has_4 and not has_3 and not has_tap and not has_drag: label = ' ⚠️ 只有type=4+其他'
        
        if label or (has_4 and has_tap):
            print(f'线 {i}: Group={line.get("Group")}, Name={line.get("Name","?"):>15s}, types={dict(sorted(types.items()))}, total={total}{label}')

# 特别看最多type=4的那条线（358个）
print('\n\n--- 有最多type=4的线详情 ---')
max_type4 = 0
max_line = None
for i, line in enumerate(lines):
    notes = line.get('notes', [])
    t4 = sum(1 for n in notes if n.get('type') == 4)
    if t4 > max_type4:
        max_type4 = t4
        max_line = (i, line)

if max_line:
    i, line = max_line
    print(f'线 {i}: Group={line.get("Group")}, Name={line.get("Name")}')
    print(f'  numOfNotes={line.get("numOfNotes")}')
    print(f'  isCover={line.get("isCover")}')
    print(f'  isGif={line.get("isGif")}')
    notes = line.get('notes', [])
    for n in notes[:3]:
        print(f'  note: type={n.get("type")}, startTime={n.get("startTime")}, posX={n.get("positionX"):.1f}, isFake={n.get("isFake")}')
    print(f'  ... (共{len(notes)}个)')
    for n in notes[-2:]:
        print(f'  note: type={n.get("type")}, startTime={n.get("startTime")}, posX={n.get("positionX"):.1f}, isFake={n.get("isFake")}')
    
    # Check if this line is at a special zOrder
    print(f'  zOrder={line.get("zOrder")}')
    print(f'  alphaControl={line.get("alphaControl")}')
    print(f'  anchor={line.get("anchor")}')
