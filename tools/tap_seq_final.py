# -*- coding: utf-8 -*-
"""最终正确统计: 直接按 tap 序列的相邻间隔 (保留多押0, 用tick判断分音)"""
import os, sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes

def tap_interval_stats(path):
    """转换后 tap 音符的相邻间隔, 按每个间隔的BPM算分音"""
    with open(path, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    all_notes, jls, bpm_tl = collect_all_notes(cd)
    times = np.array([n['time'] for n in all_notes])
    types = np.array([n['type'] for n in all_notes])
    bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
    # 只 tap
    tap = types == 1
    ts = times[tap]; bps = bpm_arr[tap]
    o = np.argsort(ts); ts = ts[o]; bps = bps[o]
    its_tick = np.diff(ts)
    bps_pair = bps[1:]
    # 分音: 每拍=32tick; 16分=8tick, 24分=5.333tick, 8分=16tick
    # 但BPM不同, 用秒: tick/32*60/bpm
    secs = its_tick / 32.0 * 60.0 / bps_pair
    thr24 = 60.0 / (bps_pair * 6)
    thr16 = 60.0 / (bps_pair * 4)
    thr8 = 60.0 / (bps_pair * 2)
    n24 = np.sum(secs < thr24)
    n16 = np.sum((secs >= thr24) & (secs < thr16))
    n8 = np.sum((secs >= thr16) & (secs < thr8))
    n_rest = np.sum(secs >= thr8)
    tot = len(secs)
    # 多押(0 tick)单独统计
    n_multi = np.sum(its_tick == 0)
    return tot, n24, n16, n8, n_rest, n_multi, its_tick

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
cases = [
    ('Melodiniq tap', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')),
    ('Verrückt tap', os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')),
    ('夢降日 tap', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')),
    ('DerSchneid tap', os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json')),
]
print(f'{"谱":<16}{"间隔":>6}{"24分":>6}{"16分":>6}{"8分":>7}{"更宽":>6}{"多押0":>6}')
for nm, p in cases:
    tot, n24, n16, n8, nr, nm_, its = tap_interval_stats(p)
    print(f'{nm:<16}{tot:>6}{n24:>6}{n16:>6}{n8:>7}{nr:>6}{nm_:>6}')
    print(f'  tick间隔: min={its.min():.0f} P25={np.percentile(its,25):.0f} P50={np.percentile(its,50):.0f}')
print('DONE')