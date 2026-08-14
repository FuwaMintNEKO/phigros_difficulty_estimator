# -*- coding: utf-8 -*-
"""Melodiniq 尾杀段分析: 最后20%的密度/位移/24分"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes, time_to_seconds

p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
all_notes, jls, bpm_tl = collect_all_notes(cd)
times = np.array([n['time'] for n in all_notes])
pos = np.array([n.get('positionX', 0) for n in all_notes])
jl = np.array([n['judge_line_idx'] for n in all_notes])
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])
dur = t_sec.max()
print(f'时长: {dur:.1f}s, 音符: {len(times)}')

# 分段: 前80% vs 尾20%
tail_mask = t_sec >= dur * 0.8
head_mask = ~tail_mask
for nm, m in [('前80%', head_mask), ('尾20%', tail_mask)]:
    n = m.sum()
    seg_t = t_sec[m]
    seg_pos = pos[m]
    seg_jl = jl[m]
    # 密度
    dens = n / max(seg_t.max() - seg_t.min(), 0.01)
    # 位移/秒
    if n > 1:
        idx = np.argsort(seg_t)
        st = seg_t[idx]; sp = seg_pos[idx]
        mov = np.abs(np.diff(sp)).sum() / max(st[-1]-st[0], 0.01)
    else:
        mov = 0
    # 24分间隔
    its = np.diff(st)
    n24 = np.sum(its < 0.013)
    print(f'{nm}: n={n} 密度={dens:.2f}/s 位移={mov:.1f}/s 24分间隔={n24}')
# 尾杀最密集 10 秒
w = 10.0
best = 0; best_t = 0
for t0 in np.arange(0, dur - w, 1.0):
    cnt = np.sum((t_sec >= t0) & (t_sec < t0 + w))
    if cnt > best:
        best = cnt; best_t = t0
print(f'\n最密10秒: t={best_t:.0f}s 音符={best} ({best/10:.1f}/s)')
# 那10秒的位移
m10 = (t_sec >= best_t) & (t_sec < best_t + 10)
if m10.sum() > 1:
    idx = np.argsort(t_sec[m10])
    mv = np.abs(np.diff(pos[m10][idx])).sum()
    print(f'  位移={mv:.1f} ({mv/10:.1f}/s)')
    its = np.diff(t_sec[m10][idx])
    print(f'  24分间隔: {np.sum(its<0.013)}')
print('DONE')