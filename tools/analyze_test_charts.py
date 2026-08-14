import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import json
import numpy as np

for name, path in [
    ('Chart_SP', os.path.join(_ROOT, 'data', 'chart', 'Chart_SP.json')),
    ('Regrets', os.path.join(_ROOT, 'data', 'chart', 'Sigma (Haocore Mix) ~ Regrets of The Yellow Tuli.json')),
    ('105秒', os.path.join(_ROOT, 'data', 'chart', 'Sigma (Haocore Mix) ~ 105秒の伝說 ~.json')),
]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    lines = data.get('judgeLineList', [])
    
    # Collect notes with type info
    tap_notes, drag_notes, hold_notes, flick_notes = [], [], [], []
    for line in lines:
        bpm = line.get('bpm', 120)
        for note in line.get('notesAbove', []) + line.get('notesBelow', []):
            t = note.get('type', 0)
            note['bpm'] = bpm
            if t == 1: tap_notes.append(note)
            elif t == 2: drag_notes.append(note)
            elif t == 3: hold_notes.append(note)
            elif t == 4: flick_notes.append(note)
    
    total = len(tap_notes) + len(drag_notes) + len(hold_notes) + len(flick_notes)
    positions = [n.get('positionX', 0) for n in tap_notes + drag_notes + hold_notes + flick_notes]
    tap_pos = [n.get('positionX', 0) for n in tap_notes]
    
    # Multi-finger (only tap+hold+flick, exclude drag)
    from collections import defaultdict
    threshold = 0.03125
    windows = defaultdict(list)
    all_notes_sorted = sorted(tap_notes + hold_notes + flick_notes, key=lambda x: x['time'])
    for note in all_notes_sorted:
        tk = round(note['time'] / threshold) * threshold
        windows[tk].append(note)
    
    true_multi_finger = sum(1 for notes in windows.values() if len(notes) >= 3)
    true_max = max(len(notes) for notes in windows.values()) if windows else 0
    
    # Hold lock analysis
    hold_ranges = [(n['time'], n['time'] + n.get('holdTime', 0), n.get('positionX', 0)) for n in hold_notes]
    lock_tap_events = 0
    move_during_lock = []
    for h_start, h_end, h_pos in hold_ranges:
        for n in tap_notes + flick_notes:
            if h_start <= n['time'] <= h_end:
                lock_tap_events += 1
                move_during_lock.append(abs(n.get('positionX', 0) - h_pos))
    
    # Fixed track: check if positions cluster on discrete values
    if tap_notes:
        tap_times = np.array([n['time'] for n in tap_notes])
        tap_positions = np.array([n.get('positionX', 0) for n in tap_notes])
        
        # 1-beat windows - check position consistency
        beat_windows = defaultdict(list)
        for n in tap_notes + hold_notes + flick_notes:
            beat_key = round(n['time'])
            beat_windows[beat_key].append(n)
        
        track_sections = 0
        for beat, notes in beat_windows.items():
            pos = [n.get('positionX', 0) for n in notes]
            if len(pos) >= 3:
                # Check if positions are within a few distinct values
                unique_p = len(set(round(p, 1) for p in pos))
                count_per_unique = [sum(1 for p in pos if abs(p - up) < 0.05) for up in set(round(p, 1) for p in pos)]
                max_on_lane = max(count_per_unique) if count_per_unique else 0
                if unique_p <= 4 and max_on_lane >= len(pos) * 0.7:
                    track_sections += 1
    
    print(f'\n{"="*50}')
    print(f'{name}')
    print(f'{"="*50}')
    print(f'  Tap: {len(tap_notes)}, Drag: {len(drag_notes)}, Hold: {len(hold_notes)}, Flick: {len(flick_notes)}')
    print(f'  总notes: {total}')
    print(f'  positionX范围: [{min(positions):.2f}, {max(positions):.2f}]')
    print(f'')
    print(f'  真·多押(不含Drag): 最大={true_max}键, 3+事件={true_multi_finger}')
    print(f'  含Drag的多押(旧算法): ', end='')
    old_windows = defaultdict(list)
    for note in sorted(tap_notes + drag_notes + hold_notes + flick_notes, key=lambda x: x['time']):
        tk = round(note['time'] / threshold) * threshold
        old_windows[tk].append(note)
    old_mf = sum(1 for notes in old_windows.values() if len(notes) >= 3)
    old_max = max(len(notes) for notes in old_windows.values()) if old_windows else 0
    print(f'最大={old_max}, 3+事件={old_mf}')
    print(f'  差异: 3+事件减少 {(old_mf - true_multi_finger)/old_mf*100:.0f}%')
    print(f'')
    print(f'  锁手分析:')
    print(f'    长条数: {len(hold_notes)}')
    print(f'    长条期间点键事件: {lock_tap_events}')
    print(f'    锁手期间位移均值: {np.mean(move_during_lock):.2f}' if move_during_lock else '    无')
    print(f'    锁手期间位移最大: {np.max(move_during_lock):.2f}' if move_during_lock else '')
    print(f'')
    print(f'  定轨段落数(近似): {track_sections}')
