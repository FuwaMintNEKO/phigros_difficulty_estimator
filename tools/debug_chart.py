import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import json

path = os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json')
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print('文件顶层键:', list(data.keys()))
lines = data.get('judgeLineList', [])
print(f'judgeLineList条目数: {len(lines)}')

if lines:
    first_line = lines[0]
    print(f'第一条线键:', list(first_line.keys()))
    print(f'  bpm: {first_line.get("bpm")}')
    
    above = first_line.get('notesAbove', [])
    below = first_line.get('notesBelow', [])
    print(f'  notesAbove: {len(above)}个, notesBelow: {len(below)}个')
    
    first_note = above[0] if above else (below[0] if below else None)
    if first_note:
        print(f'  第一个note: {first_note}')

# Check if any line has notes
total_notes = 0
empty_lines = 0
for i, line in enumerate(lines):
    na = len(line.get('notesAbove', []))
    nb = len(line.get('notesBelow', []))
    total_notes += na + nb
    if na + nb == 0:
        empty_lines += 1

print(f'\n总音符数: {total_notes}')
print(f'空线数: {empty_lines}')

# Check for speed events
total_speed = 0
for line in lines:
    total_speed += len(line.get('speedEvents', []))
print(f'总speedEvents: {total_speed}')
