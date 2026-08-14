# -*- coding: utf-8 -*-
"""speedEvents 量纲对比: 官谱 vs RPE高仿"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
# 官谱 speedEvents
p_off = os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')
raw_off = json.load(open(p_off, encoding='utf-8'))
vals_off = []
for jl in raw_off.get('judgeLineList', []):
    for ev in jl.get('speedEvents', []):
        vals_off.append(ev.get('value'))
print(f'官谱 speedEvents value: {sorted(set(vals_off))[:15]}')

# RPE 高仿 speedEvents (eventLayers里)
p_fake = os.path.join(DL, '夢の降る日に', '5333883479687925.json')
raw_fake = json.load(open(p_fake, encoding='utf-8'))
vals_fake = []
for jl in raw_fake.get('judgeLineList', []):
    for layer in jl.get('eventLayers', []) or []:
        if layer:
            for ev in layer.get('speedEvents', []):
                vals_fake.append(ev.get('start'))
                vals_fake.append(ev.get('end'))
print(f'\nRPE高仿 speedEvents start/end: {sorted(set(vals_fake))[:15]}')
# RPE顶层speedEvents
vals_top = []
for jl in raw_fake.get('judgeLineList', []):
    for ev in jl.get('speedEvents', []):
        vals_top.append(ev.get('start'))
print(f'RPE高仿 顶层speedEvents: {sorted(set(vals_top))[:15]}')
print('DONE')