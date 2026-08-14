# -*- coding: utf-8 -*-
"""finger_peak_tps 差异: 高仿18 vs 官谱10 (同一个谱不该差这么多)"""
import os, sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
def finger_stats(path):
    with open(path, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    all_notes, jls, bpm_tl = collect_all_notes(cd)
    times = np.array([n['time'] for n in all_notes])
    types = np.array([n['type'] for n in all_notes])
    pos = np.array([n.get('positionX', 0) for n in all_notes])
    bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
    # 全音符
    o = np.argsort(times)
    ts, ty, tp, bp = times[o], types[o], pos[o], bpm_arr[o]
    # 每个1秒窗口的tps
    tsec = ts / 32.0 * 60.0 / np.maximum(bp, 1.0)
    dur = tsec.max()
    best = 0
    for t0 in np.arange(0, dur-1, 0.25):
        c = np.sum((tsec>=t0)&(tsec<t0+1))
        if c > best: best = c
    return best, dur

p1 = os.path.join(DL, '夢の降る日に', '5333883479687925.json')
p2 = os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')
b1, d1 = finger_stats(p1)
b2, d2 = finger_stats(p2)
print(f'高仿: 峰值全音符tps={b1}/s (时长{d1:.0f}s)')
print(f'官谱: 峰值全音符tps={b2}/s (时长{d2:.0f}s)')
print(f'\n高仿finger_peak_tps=18 vs 官谱=10 — 全音符峰值差1.8倍')
print('可能finger特征只算某类型? 检查')
print('DONE')