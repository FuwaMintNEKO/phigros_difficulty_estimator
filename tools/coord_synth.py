# -*- coding: utf-8 -*-
"""Phigros坐标系统解析: 判定线位置与音符位置的合成"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# 看 RPE 的 yOffset/father 等 + 官谱判定线在音符时刻的位置
p = os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')
raw = json.load(open(p, encoding='utf-8'))
jls = raw.get('judgeLineList', [])
line0 = jls[0]
# 线0 在 128tick 时的位置 (128tick=1拍@238bpm)
evs = line0.get('judgeLineMoveEvents', [])
# 找覆盖 t=128 的事件
for ev in evs:
    st, et = ev['startTime'], ev['endTime']
    if st <= 128 <= et:
        # 线性插值
        if et == st: pos = ev['start']
        else:
            r = (128 - st) / (et - st)
            pos = ev['start'] + (ev['end'] - ev['start']) * r
        print(f'线0 @128tick: x={pos:.3f}')
        break
# 音符@128tick 附近的 positionX
notes = line0.get('notesAbove', [])
for n in notes[:5]:
    print(f'  音符 t={n["time"]} positionX={n["positionX"]}')
# 结论: 音符屏幕位置 = 判定线x + 音符positionX?
# 验证: 判定线x范围±5, 音符positionX范围±7.66 → 但官方音符positionX是-9~9?
# 看RPE的yOffset - 判定线在y方向的位置
print('\n=== 关键: 标准判定线位置 -999999 (初始) ===')
for ev in line0.get('judgeLineMoveEvents', [])[:3]:
    print(f'  st={ev["startTime"]} et={ev["endTime"]} x: {ev["start"]}→{ev["end"]} y: {ev["start2"]}→{ev["end2"]}')
print('DONE')