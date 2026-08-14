# -*- coding: utf-8 -*-
"""核心实验: 音符屏幕绝对位置位移 vs 局部positionX位移"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes, time_to_seconds, _parse_bpm_timeline

def analyze(path):
    with open(path, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    jls = cd.get('judgeLineList', [])
    bpm_tl = _parse_bpm_timeline(cd)
    all_notes, _, bpm_tl2 = collect_all_notes(cd)
    core_mask = np.array([n['type'] in (1,3) for n in all_notes])
    t_sec = np.array([time_to_seconds(n['time'], max(n.get('bpm',120),1.0), bpm_tl2) for n in all_notes])
    times = np.array([n['time'] for n in all_notes])
    jl_idx = np.array([n['judge_line_idx'] for n in all_notes])
    pos_x = np.array([n.get('positionX', 0) for n in all_notes])
    # 判定线屏幕x(t): 每线构建插值函数
    line_pos = {}  # line_idx -> (t_list, x_list)
    for li, jl in enumerate(jls):
        moves = jl.get('judgeLineMoveEvents', [])
        if not moves:
            line_pos[li] = (np.array([-1e9, 1e9]), np.array([0.0, 0.0]))
            continue
        ts_ = []; xs_ = []
        for ev in moves:
            st, et = ev['startTime'], ev['endTime']
            ts_.extend([st, et]); xs_.extend([ev['start'], ev['end']])
        # 去重排序
        ts_ = np.array(ts_); xs_ = np.array(xs_)
        o = np.argsort(ts_)
        ts_, xs_ = ts_[o], xs_[o]
        # 去重
        uniq = np.unique(ts_, return_index=True)[1]
        line_pos[li] = (ts_[uniq], xs_[uniq])
    # 屏幕位置 = 判定线x(t) + positionX
    screen = np.zeros(len(all_notes))
    for i in range(len(all_notes)):
        li = jl_idx[i]; t = times[i]
        tlist, xlist = line_pos.get(li, (np.array([-1e9,1e9]), np.array([0.,0.])))
        x = np.interp(t, tlist, xlist, left=xlist[0], right=xlist[-1])
        screen[i] = x + pos_x[i]
    # 位移 (core, 时间排序)
    o = np.argsort(t_sec)
    sc = screen[core_mask][o[core_mask[o]]] if False else None
    idx = np.where(core_mask)[0]
    idx = idx[np.argsort(t_sec[idx])]
    sc = screen[idx]
    lc = pos_x[idx]
    dur = t_sec[idx].max() - t_sec[idx].min()
    mov_screen = np.abs(np.diff(sc)).sum() / dur
    mov_local = np.abs(np.diff(lc)).sum() / dur
    return mov_screen, mov_local, len(idx), dur

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
cases = [
    ('Melodiniq', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')),
    ('Verrückt IN', os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')),
    ('夢降日 IN', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')),
    ('DerSchneid AT', os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json')),
]
print(f'{"谱":<18}{"屏幕位移/s":>12}{"局部位移/s":>12}{"屏幕/局部":>10}{"core数":>8}')
for nm, p in cases:
    try:
        ms, ml, n, dur = analyze(p)
        print(f'{nm:<18}{ms:>12.1f}{ml:>12.1f}{ms/max(ml,0.01):>10.2f}{n:>8}')
    except Exception as e:
        print(f'{nm}: ERR {e}')
print('DONE')