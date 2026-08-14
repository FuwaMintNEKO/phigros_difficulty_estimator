# -*- coding: utf-8 -*-
"""核心实验: 音符屏幕绝对位置位移 vs 局部positionX位移 (Melodiniq vs 官谱)"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes, time_to_seconds

def screen_positions(path, core_only=True):
    """计算每个音符的屏幕绝对位置"""
    with open(path, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    if isinstance(cd, dict):
        raw = cd
    else:
        return None
    jls = raw.get('judgeLineList', [])
    bpm_tl = _parse_bpm_tl(raw)
    all_items = []
    for li, jl in enumerate(jls):
        # 判定线 moveEvents → 分段线性位置函数
        moves = jl.get('judgeLineMoveEvents', [])
        # 线初始位置: 第一个事件之前用其 start
        for n in jl.get('notesAbove', []) + jl.get('notesBelow', []):
            t = n['time']
            # 判定线在 t 时刻的 x 位置
            lx = 0.0
            found = False
            for ev in moves:
                st, et = ev['startTime'], ev['endTime']
                if st <= t <= et or (not found and et >= t):
                    if et == st: lx = ev['start']
                    else:
                        r = (t - st) / (et - st)
                        lx = ev['start'] + (ev['end'] - ev['start']) * r
                    found = True
                    break
            if not found and moves:
                lx = moves[0]['start']
            if core_only and n['type'] not in (1, 3): continue
            all_items.append((t, lx + n.get('positionX', 0), n['type'], li))
    all_items.sort(key=lambda x: x[0])
    return all_items, bpm_tl

def _parse_bpm_tl(cd):
    from feature_extractor import _parse_bpm_timeline
    return _parse_bpm_timeline(cd)

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
cases = [
    ('Melodiniq', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')),
    ('Verrückt IN', os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')),
    ('夢降日 IN', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')),
]
for nm, p in cases:
    res = screen_positions(p)
    if res is None: continue
    items, bpm_tl = res
    ts = np.array([x[0] for x in items])
    sp = np.array([x[1] for x in items])
    # 秒
    bpm_arr = np.array([240.0]*len(ts))  # 近似
    # 用 collect_all_notes 拿真实 bpm
    with open(p, 'rb') as f:
        cd2, _ = load_chart_from_bytes(f.read())
    all_notes, _, bpm_tl2 = collect_all_notes(cd2)
    t_sec = np.array([time_to_seconds(n['time'], max(n.get('bpm',120),1.0), bpm_tl2) for n in all_notes])
    # 只留 core
    core_mask = np.array([n['type'] in (1,3) for n in all_notes])
    tc = t_sec[core_mask]
    # 屏幕位移 (相邻core音符)
    o = np.argsort(tc)
    # 屏幕位置按时间排序
    sp_sorted = sp[np.argsort(ts)]
    mov = np.abs(np.diff(sp_sorted)).sum() / max(tc.max()-tc.min(), 0.01)
    # 局部位移 (用 positionX 只差)
    local_pos = np.array([x[1] - (判定线未知) for x in items])
    # 简化: 直接算局部
    with open(p, 'rb') as f:
        cd3, _ = load_chart_from_bytes(f.read())
    all_notes3, _, _ = collect_all_notes(cd3)
    lpos = np.array([n.get('positionX', 0) for n in all_notes3])
    cm3 = np.array([n['type'] in (1,3) for n in all_notes3])
    lpos_c = lpos[cm3][np.argsort(tc)]
    lmov = np.abs(np.diff(lpos_c)).sum() / max(tc.max()-tc.min(), 0.01)
    print(f'{nm}: 屏幕位移={mov:.1f}/s  局部位移={lmov:.1f}/s  屏幕/局部={mov/max(lmov,0.01):.2f}x')
print('DONE')