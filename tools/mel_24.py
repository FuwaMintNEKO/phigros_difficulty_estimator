# -*- coding: utf-8 -*-
"""Melodiniq 24分音符精确定位: 哪些段有24分, 对应BPM"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
raw = json.load(open(p, encoding='utf-8'))
# BPM 时间线 (ticks)
bpms = raw.get('BPMList', [])
bpm_events = []
for b in bpms:
    st = b.get('startTime')
    if isinstance(st, list) and len(st) == 3:
        t = (float(st[0])*4.0 + float(st[1])*(4.0/float(st[2]))) * 8.0
        bpm_events.append((t, float(b['bpm'])))
bpm_events.sort()
print('BPM时间线:', [(round(t,1), b) for t, b in bpm_events])
# 音符时间
notes = []
for jl in raw.get('judgeLineList', []):
    for n in jl.get('notes', []):
        if int(n.get('isFake', 0) or 0) == 1: continue
        st = n.get('startTime')
        if isinstance(st, list) and len(st) == 3:
            t = (float(st[0])*4.0 + float(st[1])*(4.0/float(st[2]))) * 8.0
            notes.append(t)
notes.sort()
notes = np.array(notes)
intervals = np.diff(notes)
print(f'\n音符数={len(notes)} 时长={notes.max()/8/4:.1f}拍')
# BPM@ticks: 每拍=32ticks, 每tick时长 = 60/BPM/8 (8分音符=4拍? 需要核对)
# Phigros: 1拍=32 ticks? 之前 K=32 (拍→ticks)
# 24分音符间隔 = 32/24 = 1.333 ticks; 在BPM B下毫秒 = 1.333/32 * 60/B * 1000
# 找 24分 (间隔<=1.4 ticks) 的簇
idx24 = np.where(intervals <= 1.4)[0]
print(f'24分间隔(<=1.4t)数量: {len(idx24)}')
if len(idx24):
    # 这些间隔的时间位置
    t24 = notes[idx24]
    print('24分簇时间范围:', round(t24.min()/8/4,1), '-', round(t24.max()/8/4,1), '拍')
    # 对应BPM
    for t0, b in bpm_events:
        # 找 t24 所在BPM段
        pass
    # 每段的BPM
    for i in range(len(bpm_events)-1):
        seg_t0, seg_b = bpm_events[i]
        seg_t1 = bpm_events[i+1][0]
        n_in = np.sum((t24 >= seg_t0) & (t24 < seg_t1))
        if n_in:
            print(f'  BPM {seg_b}: 24分间隔 {n_in} 个 (等效16分BPM = {seg_b*24/16:.0f})')
# 尾杀: 最后20% 的密度和位移
tail = notes[notes >= notes.max()*0.8]
print(f'\n尾杀(最后20%): {len(tail)} 音符, 间隔P25={np.percentile(np.diff(tail),25):.1f} ticks')
print('DONE')