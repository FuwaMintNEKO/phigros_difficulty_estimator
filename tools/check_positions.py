import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from data_loader import load_chart_json
import numpy as np

path = os.path.join(_ROOT, 'data', 'chart', 'Rrharil.TeamGrimoire.0', 'AT.json')
data = load_chart_json(path)
all_notes = []
for line in data.get('judgeLineList', []):
    for note in line.get('notesAbove', []) + line.get('notesBelow', []):
        note['bpm'] = line.get('bpm', 120)
        all_notes.append(note)

positions = [n.get('positionX', 0) for n in all_notes]
types = [n.get('type', 0) for n in all_notes]

print(f'总notes: {len(all_notes)}')
print(f'positionX 范围: {min(positions):.3f} ~ {max(positions):.3f}')
print(f'positionX 均值: {np.mean(positions):.3f} ± {np.std(positions):.3f}')

# 离散值分析
unique_pos = sorted(set(round(p, 3) for p in positions))
print(f'唯一位置数: {len(unique_pos)}')
print(f'分布: {unique_pos[:20]}...')

# 看看不同类型的position
for t in [1, 2, 3, 4]:
    p = [positions[i] for i in range(len(all_notes)) if types[i] == t]
    if p:
        print(f'\ntype={t} ({["","Tap","Drag","Hold","Flick"][t-1] if t<=4 else "?"}): range=[{min(p):.3f}, {max(p):.3f}], mean={np.mean(p):.3f}')
