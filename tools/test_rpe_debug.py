import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

CHART_PATH = os.path.join(_ROOT, 'data', 'chart', '1321664301929799.json')

with open(CHART_PATH, 'r', encoding='utf-8') as f:
    raw = json.load(f)

lines = raw.get('judgeLineList', [])
meta = raw.get('META', {})
print(f'谱面: {meta.get("name")}')
print(f'META duration: {meta.get("duration")}s')
print(f'BPM: {raw.get("BPMList", [{}])[0].get("bpm")}')
print(f'judgeLineList: {len(lines)}条\n')

total_notes = 0
min_time = float('inf')
max_time = float('-inf')
time_ranges = []
no_notes_lines = 0
has_notes_display = 0
has_notes = 0

for i, line in enumerate(lines):
    notes = line.get('notes', [])
    notes_display = line.get('notes_display', [])
    
    if notes:
        has_notes += 1
    if notes_display:
        has_notes_display += 1
    
    src = notes if notes else notes_display
    
    if not src:
        no_notes_lines += 1
        continue
    
    for n in src:
        total_notes += 1
        st = n.get('startTime', [0,0,1])
        if isinstance(st, list) and len(st) >= 3:
            measure, beat, div = st[0], st[1], st[2]
            # convert to beat units same as converter
            t_beats = float(measure) * 4.0 + float(beat) * (4.0 / float(div))
            min_time = min(min_time, t_beats)
            max_time = max(max_time, t_beats)

print(f'使用"notes"字段的line: {has_notes}')
print(f'使用"notes_display"字段的line: {has_notes_display}')
print(f'空line: {no_notes_lines}')
print(f'总note数: {total_notes}')
print(f'首note(beats): {min_time if min_time != float("inf") else "N/A"}')
print(f'尾note(beats): {max_time if max_time != float("-inf") else "N/A"}')
if max_time > 0:
    bpm = raw.get("BPMList", [{}])[0].get("bpm", 120)
    duration_sec = max_time / bpm * 1.875
    print(f'计算时长: {duration_sec:.2f}秒 ({max_time:.0f}拍 @ {bpm}BPM)')

# Check a sample note to see structure
for i, line in enumerate(lines):
    notes = line.get('notes', []) or line.get('notes_display', [])
    if notes:
        print(f'\nLine {i} 第一条note: {json.dumps(notes[0], ensure_ascii=False)}')
        break

# Check if there are speedEvents
total_speed = 0
for i, line in enumerate(lines):
    se = line.get('speedEvents', [])
    if se:
        total_speed += len(se)
print(f'\n总speedEvents: {total_speed}')
