# -*- coding: utf-8 -*-
"""综合对比: Melodiniq vs 官谱16.5+ 的 tap 密度/位移/多押"""
import os, sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes, time_to_seconds

def stats(path):
    with open(path, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    all_notes, jls, bpm_tl = collect_all_notes(cd)
    times = np.array([n['time'] for n in all_notes])
    types = np.array([n['type'] for n in all_notes])
    pos = np.array([n.get('positionX', 0) for n in all_notes])
    bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
    t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])
    dur = t_sec.max()
    o = np.argsort(t_sec)
    ts, ty, tp = t_sec[o], types[o], pos[o]
    tap = ty == 1
    # tap密度
    tap_dens = tap.sum() / dur
    # tap 最密20秒
    tt = ts[tap]; tpp = tp[tap]
    best = 0; bt = 0
    for t0 in np.arange(0, dur-20, 0.5):
        c = np.sum((tt>=t0)&(tt<t0+20))
        if c > best: best, bt = c, t0
    m = (tt>=bt)&(tt<bt+20)
    mov = np.abs(np.diff(tpp[m])).sum()/20
    its = np.diff(tt[m])
    p50 = np.percentile(its,50)*1000
    # 多押
    mof = np.sum(np.diff(ts) < 1e-6)
    # drag比
    drag_ratio = np.sum(ty==2)/len(ty)
    return {'dur': dur, 'tap_dens': tap_dens, 'peak20_tap': best/20, 'peak20_mov': mov,
            'peak20_p50': p50, 'multi_press': mof, 'drag_ratio': drag_ratio, 'n': len(ty)}

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
for nm, p in [
    ('Melodiniq', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')),
    ('Verrückt(16.5)', os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')),
    ('夢降日(16.6)', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')),
    ('DerSchneid(17.5)', os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json')),
]:
    s = stats(p)
    print(f'{nm:<18} dur={s["dur"]:.0f}s tap密度={s["tap_dens"]:.2f}/s 峰值20s={s["peak20_tap"]:.1f}/s 峰值位移={s["peak20_mov"]:.0f}/s 峰值P50={s["peak20_p50"]:.0f}ms 多押={s["multi_press"]} drag比={s["drag_ratio"]:.2f}')
print('DONE')