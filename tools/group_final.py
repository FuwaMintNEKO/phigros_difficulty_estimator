# -*- coding: utf-8 -*-
"""最终正确统计: 按多押组(同tick合并)计算 tap+hold 分音
关键: 多押=同一时刻, 组间间隔才是真实节奏"""
import os, sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes

def group_intervals(path, core_only=True, include_drag=False):
    with open(path, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    all_notes, jls, bpm_tl = collect_all_notes(cd)
    times = np.array([n['time'] for n in all_notes])
    types = np.array([n['type'] for n in all_notes])
    bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
    if include_drag:
        mask = np.ones(len(types), dtype=bool)
    else:
        mask = (types == 1) | (types == 3)
    # 按tick分组 (同一tick=同一时刻, 合并为多押组)
    grp = {}
    for i in range(len(times)):
        if mask[i]:
            t = times[i]
            grp.setdefault(t, []).append(bpm_arr[i])
    gtimes = np.array(sorted(grp.keys()))
    gbpms = np.array([max(grp[t]) for t in gtimes])  # 组BPM用最大的
    gits = np.diff(gtimes)  # tick
    secs = gits / 32.0 * 60.0 / gbpms[1:]
    thr24 = 60.0 / (gbpms[1:] * 6)
    thr16 = 60.0 / (gbpms[1:] * 4)
    thr8 = 60.0 / (gbpms[1:] * 2)
    n24 = np.sum(secs < thr24)
    n16 = np.sum((secs >= thr24) & (secs < thr16))
    n8 = np.sum((secs >= thr16) & (secs < thr8))
    n_rest = np.sum(secs >= thr8)
    return len(gtimes), n24, n16, n8, n_rest, gits

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
cases = [
    ('Melodiniq (tap+hold)', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json'), False),
    ('Melodiniq (含drag)', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json'), True),
    ('Verrückt (tap+hold)', os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json'), False),
    ('夢降日 (tap+hold)', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json'), False),
    ('DerSchneid (tap+hold)', os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json'), False),
]
print(f'{"谱":<24}{"组数":>6}{"24分":>6}{"16分":>6}{"8分":>6}{"更宽":>6}')
for nm, p, inc in cases:
    ng, n24, n16, n8, nr, _ = group_intervals(p, include_drag=inc)
    print(f'{nm:<24}{ng:>6}{n24:>6}{n16:>6}{n8:>6}{nr:>6}')
print('DONE')