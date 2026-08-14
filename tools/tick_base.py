# -*- coding: utf-8 -*-
"""Melodiniq 时间单位彻底验证: BPMList 的 tick 基准"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
raw = json.load(open(p, encoding='utf-8'))
bpms = raw.get('BPMList', [])
print('BPMList:')
for b in bpms[:15]:
    print(f'  bpm={b.get("bpm")} startTime={b.get("startTime")}')
# 关键: RPE 的 startTime [m,b,d] → 拍 → tick
# 官方 tick 基准: 1拍=32 ticks? 还是 1拍=1?
# 看官谱: 音符 time 的典型间隔
p2 = os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')
raw2 = json.load(open(p2, encoding='utf-8'))
notes2 = []
for jl in raw2.get('judgeLineList', []):
    for n in jl.get('notesAbove', []):
        notes2.append(n['time'])
notes2.sort()
its2 = np.diff(np.array(notes2))
print(f'\n官谱 Verrückt 音符time间隔: min={its2.min():.2f} P25={np.percentile(its2,25):.2f} P50={np.percentile(its2,50):.2f}')
print(f'  16分音符间隔应该是2 (1拍=32tick) 或 0.5 (1拍=1tick)')
# Melodiniq RPE 转换后的 time
p3 = p
with open(p3, 'rb') as f:
    from unified_parser import load_chart_from_bytes
    cd, _ = load_chart_from_bytes(f.read())
notes3 = []
for jl in cd.get('judgeLineList', []):
    for n in jl.get('notesAbove', []) + jl.get('notesBelow', []):
        notes3.append(n['time'])
notes3.sort()
its3 = np.diff(np.array(notes3))
print(f'\nMelodiniq 转换后音符time间隔: min={its3.min():.2f} P25={np.percentile(its3,25):.2f} P50={np.percentile(its3,50):.2f} P75={np.percentile(its3,75):.2f}')
print(f'  如果P50=8: 1拍=32tick → 8tick=4分音符? 或 1拍=1tick → 8tick=8拍?')
# 验证: RPE startTime [m,b,d], m=小节 b=拍 d=等分
# 1小节=4拍 → beat = m*4 + b*(4/d)? 我的公式
# Phigros官方: beat = m*4 + b + d? 
# 看 BPM 事件时间: 193bpm@[0,0,1], 196bpm@[256,0,1]
# 如果[256,0,1] = 256小节 = 1024拍, 那BPM变化在1024拍处
# 但之前显示 bpm_events tick: (8192, 196) → 8192tick = 1024拍 → 1拍=8tick!
print('\n关键: 8192 tick = 256小节*4拍 = 1024拍 → 1拍 = 8192/1024 = 8 ticks!')
print('也就是说 RPE 时间: 1拍=8 tick, 不是32!')
print('16分音符 = 8/4 = 2 tick ✓  24分 = 8/6 = 1.333 tick ✓')
print('32分 = 8/8 = 1 tick')
print('DONE')