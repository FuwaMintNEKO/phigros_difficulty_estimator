# -*- coding: utf-8 -*-
"""短窗口爆发对比: 1s/2s/3s/5s/10s 的 tap+hold 密度与位移峰值"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes, time_to_seconds

def short_burst(path, window):
    """tap+hold 在 window 秒窗口内的峰值密度 + 峰值位移"""
    with open(path, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    all_notes, jls, bpm_tl = collect_all_notes(cd)
    times = np.array([n['time'] for n in all_notes])
    types = np.array([n['type'] for n in all_notes])
    pos = np.array([n.get('positionX', 0) for n in all_notes])
    bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
    t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])
    dur = t_sec.max()
    core = (types == 1) | (types == 3)
    tt = t_sec[core]; tp = pos[core]
    o = np.argsort(tt); tt, tp = tt[o], tp[o]
    if len(tt) < 2:
        return 0, 0, 0
    # 峰值密度
    best = 0; bt = 0
    step = max(window/4, 0.1)
    for t0 in np.arange(0, dur-window, step):
        c = np.sum((tt>=t0)&(tt<t0+window))
        if c > best: best, bt = c, t0
    m = (tt>=bt)&(tt<bt+window)
    # 该窗口内的位移 (按时间序)
    mv = np.abs(np.diff(tp[m])).sum() / window
    # 窗口内间隔P25 (爆发速度)
    its = np.diff(tt[m])
    p25 = np.percentile(its, 25)*1000 if len(its) else 999
    return best/window, mv, p25

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
cases = [
    ('Melodiniq', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')),
    ('Verrückt(16.5)', os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')),
    ('夢降日(16.6)', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')),
    ('DerSchneid(17.5)', os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json')),
]
print('=== tap+hold 短窗口峰值密度 (/s) ===')
print(f'{"谱":<18}', end='')
for w in [1, 2, 3, 5, 10, 20]:
    print(f'{w:>6}s', end='')
print()
results = {}
for nm, p in cases:
    print(f'{nm:<18}', end='')
    row = []
    for w in [1, 2, 3, 5, 10, 20]:
        d, m, p25 = short_burst(p, w)
        row.append((d, m, p25))
        print(f'{d:>6.1f}', end='')
    print()
    results[nm] = row

print('\n=== tap+hold 短窗口峰值位移 (/s) ===')
print(f'{"谱":<18}', end='')
for w in [1, 2, 3, 5, 10, 20]:
    print(f'{w:>6}s', end='')
print()
for nm, p in cases:
    print(f'{nm:<18}', end='')
    for w in [1, 2, 3, 5, 10, 20]:
        d, m, p25 = short_burst(p, w)
        print(f'{m:>6.0f}', end='')
    print()

print('\n=== tap+hold 峰值窗口内间隔P25 (ms) ===')
print(f'{"谱":<18}', end='')
for w in [1, 2, 3, 5]:
    print(f'{w:>6}s', end='')
print()
for nm, p in cases:
    print(f'{nm:<18}', end='')
    for w in [1, 2, 3, 5]:
        d, m, p25 = short_burst(p, w)
        print(f'{p25:>6.0f}', end='')
    print()
print('DONE')