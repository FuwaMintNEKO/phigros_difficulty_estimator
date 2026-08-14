# -*- coding: utf-8 -*-
"""Melodiniq vs Verrückt: tap-only 完整对比"""
import os, sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes, time_to_seconds

def tap_stats(path):
    with open(path, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    all_notes, jls, bpm_tl = collect_all_notes(cd)
    times = np.array([n['time'] for n in all_notes])
    types = np.array([n['type'] for n in all_notes])
    pos = np.array([n.get('positionX', 0) for n in all_notes])
    bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
    t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])
    dur = t_sec.max()
    core = (types==1)|(types==3)
    tc = t_sec[core]; tp = pos[core]
    # 排序
    o = np.argsort(tc); tc, tp = tc[o], tp[o]
    its = np.diff(tc)
    # 最密10秒
    best=0; bt=0
    for t0 in np.arange(0, dur-10, 0.5):
        c = np.sum((tc>=t0)&(tc<t0+10))
        if c>best: best, bt = c, t0
    m = (tc>=bt)&(tc<bt+10)
    mv = np.abs(np.diff(tp[m])).sum()
    return {'dur': dur, 'n_core': len(tc), 'density': len(tc)/dur,
            'peak10': best/10, 'peak10_mov': mv/10,
            'its_p50': np.percentile(its,50)*1000, 'its_p25': np.percentile(its,25)*1000,
            '<50ms': np.mean(its<0.05)*100}

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
r_mel = tap_stats(os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json'))
r_ver = tap_stats(os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json'))
r_yum = tap_stats(os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json'))
print(f'{"指标":<20}{"Melodiniq":>12}{"Verrückt(16.5)":>14}{"夢降日(16.6)":>14}')
for k in r_mel:
    print(f'{k:<20}{r_mel[k]:>12.1f}{r_ver[k]:>14.1f}{r_yum[k]:>14.1f}')
print('DONE')