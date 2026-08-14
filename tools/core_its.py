# -*- coding: utf-8 -*-
"""修复后验证: Melodiniq vs 官谱16.5+ 的 tap+hold 分音特征"""
import os, sys, io, json, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features, collect_all_notes, time_to_seconds

def core_interval_stats(path):
    """tap+hold 相邻间隔 (排除多押0ms) 统计"""
    with open(path, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    all_notes, jls, bpm_tl = collect_all_notes(cd)
    times = np.array([n['time'] for n in all_notes])
    types = np.array([n['type'] for n in all_notes])
    jl_idx = np.array([n['judge_line_idx'] for n in all_notes])
    bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
    t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])
    core = (types == 1) | (types == 3)
    # 排序
    o = np.argsort(t_sec)
    ts = t_sec[o]; ty = types[o]; jl = jl_idx[o]; cm = core[o]
    its = np.diff(ts)
    adj = cm[1:] & cm[:-1] & (its > 1e-6)
    same = jl[1:] == jl[:-1]
    # 全部core相邻 (跨线也算: 手指切换)
    all_adj = its[adj]
    same_adj = its[adj & same]
    return {
        'n_its': len(all_adj),
        '<20ms(24分@240+)': np.sum(all_adj < 0.020),
        '<42ms(24分@193)': np.sum(all_adj < 0.042),
        '<63ms(16分@240)': np.sum(all_adj < 0.063),
        'P25_ms': np.percentile(all_adj, 25)*1000,
        'P50_ms': np.percentile(all_adj, 50)*1000,
        'same_ratio': len(same_adj)/max(len(all_adj),1),
        'dur': ts.max(),
    }

cases = [
    ('Melodiniq', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')),
    ('Verrückt IN', os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')),
    ('夢降日 IN', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')),
    ('DerSchneid AT', os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json')),
]
print(f'{"谱":<16}{"间隔数":>7}{"24分@240":>10}{"24分@193":>10}{"16分@240":>10}{"P25ms":>8}{"P50ms":>8}{"同线比":>8}')
for nm, p in cases:
    s = core_interval_stats(p)
    print(f'{nm:<16}{s["n_its"]:>7}{s["<20ms(24分@240+)"]:>10}{s["<42ms(24分@193)"]:>10}{s["<63ms(16分@240)"]:>10}{s["P25_ms"]:>8.0f}{s["P50_ms"]:>8.0f}{s["same_ratio"]:>8.2f}')
print('DONE')