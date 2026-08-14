import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import json
import numpy as np

path = os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json')
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

lines = data.get('judgeLineList', [])
all_notes = []

# Try different time conversions
def to_beat_time_v1(t):
    measure, beat, div = t
    return measure * 4 + beat + (div - 1) / 192.0

def to_beat_time_v2(t, tpq=960):
    measure, beat, div = t
    return measure * 4 + beat + (div - 1) / tpq

def to_beat_time_v3(t, tpq=1920):
    measure, beat, div = t
    return measure * 4 + beat + (div - 1) / tpq

def to_beat_time_onediv(t, divs_per_beat=192):
    measure, beat, div = t
    return (measure * 4 * divs_per_beat + beat * divs_per_beat + div - 1) / divs_per_beat

for line in lines:
    notes = line.get('notes', [])
    for n in notes:
        st = n.get('startTime', [0,0,1])
        all_notes.append({
            'type': n.get('type'),
            'startTime': st,
            'positionX': n.get('positionX', 0),
        })

# Sort by startTime
all_notes.sort(key=lambda x: (x['startTime'][0], x['startTime'][1], x['startTime'][2]))

print(f'总notes (RPE格式): {len(all_notes)}')

# First and last 3 notes
print(f'\n第一条notes: {all_notes[0]}')
print(f'第二条notes: {all_notes[1]}')
print(f'最后一条notes: {all_notes[-1]}')
print(f'倒数第二条notes: {all_notes[-2]}')

# Try different conversions
bpm = 180
for name, fn in [('v1_div192', to_beat_time_v1), ('v2_div960', to_beat_time_v2), 
                  ('v3_div1920', to_beat_time_v3), ('onediv', to_beat_time_onediv)]:
    first_t = fn(all_notes[0]['startTime'])
    last_t = fn(all_notes[-1]['startTime'])
    duration_b = last_t - first_t
    duration_s = (duration_b / bpm) * 1.875
    print(f'\n{name}:')
    print(f'  第一条: {all_notes[0]["startTime"]} → {first_t:.4f} beats')
    print(f'  最后一条: {all_notes[-1]["startTime"]} → {last_t:.4f} beats')
    print(f'  持续时间: {duration_b:.2f} beats = {duration_s:.2f} sec')
    print(f'  密度: {len(all_notes)/max(duration_s,0.01):.2f} notes/sec')

# Check different division values in the notes
all_divs = set()
for n in all_notes:
    all_divs.add(n['startTime'][2])
print(f'\n出现的division值: {sorted(all_divs)[:30]}')
print(f'最大division: {max(all_divs)}')

# Check measure range
all_measures = [n['startTime'][0] for n in all_notes]
print(f'measure范围: {min(all_measures)} ~ {max(all_measures)}')

# Check positionX range
pos_x = [n['positionX'] for n in all_notes]
print(f'positionX范围: {min(pos_x):.1f} ~ {max(pos_x):.1f}')
# Check unique positions
unique_pos = sorted(set(round(p, 1) for p in pos_x))
print(f'唯一positionX (前20个): {unique_pos[:20]}')
