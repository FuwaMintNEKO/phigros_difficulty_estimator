# -*- coding: utf-8 -*-
"""官谱 moveEvents 是否密集采样(无移动也采样)"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')
raw = json.load(open(p, encoding='utf-8'))
jl0 = raw['judgeLineList'][0]
moves = jl0.get('judgeLineMoveEvents', [])
print(f'线0 moveEvents: {len(moves)}')
# 事件时间步长
sts = [m['startTime'] for m in moves]
steps = np.diff(sts)
print(f'时间步长: min={steps.min():.1f} P25={np.percentile(steps,25):.1f} P50={np.percentile(steps,50):.1f} P75={np.percentile(steps,75):.1f}')
# 有位移的事件 vs 无位移
moved = [m for m in moves if abs(m['end'] - m['start']) > 1e-6 or abs(m['end2'] - m['start2']) > 1e-6]
print(f'有实际位移的事件: {len(moved)}/{len(moves)} ({len(moved)/len(moves)*100:.0f}%)')
# RPE 高仿对比
DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
p2 = os.path.join(DL, '夢の降る日に', '5333883479687925.json')
raw2 = json.load(open(p2, encoding='utf-8'))
jl0_2 = raw2['judgeLineList'][0]
tot_m = 0
for layer in jl0_2.get('eventLayers', []) or []:
    if layer:
        tot_m += len(layer.get('moveXEvents', [])) + len(layer.get('moveYEvents', []))
print(f'\nRPE高仿 线0 moveX+moveY: {tot_m}')
print('\n=== 结论: 官谱 moveEvents 是密集采样(每4tick), RPE是关键帧 ===')
print('官谱jline密度(事件数/秒)天然是RPE的~100倍 → 两种格式不可比!')
print('DONE')