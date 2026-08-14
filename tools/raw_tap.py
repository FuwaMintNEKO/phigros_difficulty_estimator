# -*- coding: utf-8 -*-
"""直接打印: Melodiniq 高潮段 tap 音符的原始startTime + 转换后time + 间隔"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes

p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
# 1) 原始 RPE
raw = json.load(open(p, encoding='utf-8'))
print('=== 原始RPE: 高潮段(拍106-126) tap音符(type=1) ===')
notes = []
for jl in raw.get('judgeLineList', []):
    for n in jl.get('notes', []):
        st = n.get('startTime')
        if isinstance(st, list) and len(st) == 3:
            # RPE官方语义: startBeat = st[0] + st[1]/st[2] (拍)
            beat = st[0] + st[1]/max(st[2],1)
            notes.append({'beat': beat, 'type': n.get('type'), 'st': st, 'et': n.get('endTime')})
notes.sort(key=lambda x: x['beat'])
# 高潮段 106-126拍
seg = [n for n in notes if 106 <= n['beat'] < 126]
taps = [n for n in seg if n['type'] == 1]
print(f'高潮段音符: {len(seg)} (tap={sum(1 for n in seg if n["type"]==1)})')
print('\n前30个 tap 音符 (beat, startTime):')
for n in taps[:30]:
    print(f'  beat={n["beat"]:.3f} startTime={n["st"]} endTime={n["et"]}')
# tap 间隔 (拍)
taps_sorted = sorted(taps, key=lambda x: x['beat'])
if len(taps_sorted) > 1:
    beats = np.array([n['beat'] for n in taps_sorted])
    dbeats = np.diff(beats)
    print(f'\ntap间隔(拍): min={dbeats.min():.3f} P25={np.percentile(dbeats,25):.3f} P50={np.percentile(dbeats,50):.3f}')
    print(f'  24分=1/6=0.167拍, 16分=1/4=0.25拍, 8分=1/2=0.5拍')
    print(f'  <=0.167拍(24分): {np.sum(dbeats<=0.167)}')
    print(f'  <=0.25拍(16分): {np.sum(dbeats<=0.25)}')
    print(f'  <=0.5拍(8分): {np.sum(dbeats<=0.5)}')
    # 具体间隔序列
    print('\ntap间隔序列(前40):')
    print(' '.join(f'{d:.2f}' for d in dbeats[:40]))
print('DONE')