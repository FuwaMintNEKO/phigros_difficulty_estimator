# -*- coding: utf-8 -*-
"""修正判定线位置插值: 音符屏幕位置 = 判定线位置(moveEvents) + positionX
问题: 之前 screen_mov2 比值=1.00 是因为判定线位置插值bug (所有线都用了相同t列表?)"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes, time_to_seconds

def screen_positions(path):
    with open(path, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    jls = cd.get('judgeLineList', [])
    all_notes, _, bpm_tl = collect_all_notes(cd)
    n = len(all_notes)
    times = np.array([x['time'] for x in all_notes])
    types = np.array([x['type'] for x in all_notes])
    pos_x = np.array([x.get('positionX', 0) for x in all_notes])
    jl_idx = np.array([x['judge_line_idx'] for x in all_notes])
    bpm_arr = np.array([x.get('bpm', 120.0) for x in all_notes])
    t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])
    # 每条线的判定线位置函数
    line_interp = {}
    for li, jl in enumerate(jls):
        moves = jl.get('judgeLineMoveEvents', [])
        if not moves:
            line_interp[li] = (np.array([-1e12, 1e12]), np.array([0.0, 0.0]))
            continue
        pts = []
        for ev in moves:
            st, et = ev['startTime'], ev['endTime']
            pts.append((st, ev['start']))
            pts.append((et, ev['end']))
        pts.sort()
        ts_ = [p[0] for p in pts]; xs_ = [p[1] for p in pts]
        # 去重
        seen = {}
        for t, x in zip(ts_, xs_):
            seen[t] = x
        ts_ = np.array(sorted(seen.keys())); xs_ = np.array([seen[t] for t in ts_])
        line_interp[li] = (ts_, xs_)
    # 屏幕位置
    screen = np.zeros(n)
    for i in range(n):
        li = jl_idx[i]; t = times[i]
        ts_, xs_ = line_interp.get(li, (np.array([-1e12,1e12]), np.array([0.,0.])))
        lx = np.interp(t, ts_, xs_, left=xs_[0], right=xs_[-1])
        screen[i] = lx + pos_x[i]
    return t_sec, types, screen, pos_x

def high20(t_sec, pos):
    dur = t_sec.max()
    best = 0; bt = 0
    for t0 in np.arange(0, dur-20, 0.5):
        m = (t_sec >= t0) & (t_sec < t0+20)
        if m.sum() > best: best, bt = m.sum(), t0
    m = (t_sec >= bt) & (t_sec < bt+20)
    return bt, m, best

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
for nm, p in [
    ('Melodiniq', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')),
    ('Verrückt IN', os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')),
    ('夢降日 IN', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')),
    ('DerSchneid AT', os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json')),
]:
    t_sec, types, screen, pos_x = screen_positions(p)
    core = (types == 1) | (types == 3)
    bt, m, best = high20(t_sec[core], screen[core])
    sc = screen[core][m]
    mov_sc = np.abs(np.diff(sc)).sum()
    lc = pos_x[core][m]
    mov_lc = np.abs(np.diff(lc)).sum()
    print(f'{nm:<18} 高潮20s: {best} core ({best/20:.1f}/s) 屏幕位移={mov_sc:.0f}({mov_sc/20:.1f}/s) 局部位移={mov_lc:.0f}({mov_lc/20:.1f}/s)')
print('DONE')