# -*- coding: utf-8 -*-
"""精确分音统计: 按拍(tick/32)严格分区
32分=1/8拍=4tick, 24分=1/6拍=5.333tick, 16分=1/4拍=8tick, 8分=1/2拍=16tick"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes

def exact_division(path):
    with open(path, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    all_notes, jls, bpm_tl = collect_all_notes(cd)
    times = np.array([n['time'] for n in all_notes])
    types = np.array([n['type'] for n in all_notes])
    tap = types == 1
    ts = np.sort(times[tap])
    d_tick = np.diff(ts)
    # 只统计 d>0 (排除多押)
    d_tick = d_tick[d_tick > 0]
    d_beat = d_tick / 32.0
    # 精确分区
    n32 = np.sum(np.abs(d_beat - 1/8) < 0.02)     # 32分
    n24 = np.sum(np.abs(d_beat - 1/6) < 0.02)     # 24分
    n16 = np.sum(np.abs(d_beat - 1/4) < 0.02)     # 16分
    n8 = np.sum(np.abs(d_beat - 1/2) < 0.02)      # 8分
    n_other = len(d_beat) - n32 - n24 - n16 - n8
    # 打印d_beat的独特值
    uniq = np.unique(np.round(d_beat, 3))
    return n32, n24, n16, n8, n_other, uniq[:20]

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
cases = [
    ('Melodiniq(RPE)', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')),
    ('Verrückt(官16.5)', os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')),
    ('夢降日(官16.6)', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')),
    ('DerSchneid(官17.5)', os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json')),
]
print(f'{"谱":<18}{"32分":>6}{"24分":>6}{"16分":>7}{"8分":>7}{"其他":>6}')
for nm, p in cases:
    n32, n24, n16, n8, no, uniq = exact_division(p)
    print(f'{nm:<18}{n32:>6}{n24:>6}{n16:>7}{n8:>7}{no:>6}')
    print(f'  独特拍间隔: {uniq}')
print('DONE')