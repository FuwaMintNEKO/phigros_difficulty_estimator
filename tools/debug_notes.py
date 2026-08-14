import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import json

path = os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json')
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

lines = data.get('judgeLineList', [])

for i, line in enumerate(lines):
    if line.get('numOfNotes', 0) > 0:
        notes = line.get('notes', [])
        print(f'线 {i}: notes={len(notes)}条')
        print(f'第一条note: {notes[0]}')
        print(f'第二条note: {notes[1]}')
        print(f'类型集合: {set(n.get("type", "?") for n in notes)}')
        
        # Check isFake
        is_fake = [n.get('isFake', 0) for n in notes]
        print(f'isFake 非零: {sum(1 for f in is_fake if f)}')
        
        # Sample some notes
        for j in [0, 1, 2, 100, 500, -1]:
            n = notes[j]
            print(f'  notes[{j}]: type={n.get("type")}, time={n.get("time")}, positionX={n.get("positionX")}, holdTime={n.get("holdTime",0)}, speed={n.get("speed")}, isFake={n.get("isFake",0)}')
        
        # Check time format - is it [measure, beat, division] or raw?
        if isinstance(notes[0].get('time'), list):
            print(f'time format is [measure, beat, division]!')
        else:
            print(f'time format is raw number')
        
        break
