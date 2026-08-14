# -*- coding: utf-8 -*-
"""多押分析: 同时按下的音符数峰值 (Melodiniq 特色)"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes, time_to_seconds

def chord_stats(path):
    """同时(tap+hold)音符数: 峰值/分布"""
    with open(path, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    all_notes, jls, bpm_tl = collect_all_notes(cd)
    times = np.array([n['time'] for n in all_notes])
    types = np.array([n['type'] for n in all_notes])
    bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
    t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_tl) for t, b in zip(times, bpm_arr)])
    core = (types == 1) | (types == 3)
    tc = t_sec[core]
    # 同时窗口: 10ms内视为同时
    tc = np.sort(tc)
    # 统计每个时刻的并发数
    from collections import Counter
    # 四舍五入到10ms
    tq = np.round(tc / 0.01) * 0.01
    cnt = Counter(tq.tolist())
    chord_sizes = list(cnt.values())
    cs = np.array(chord_sizes)
    # 多押事件数 (并发>=2 的时刻)
    multi_events = np.sum(cs >= 2)
    multi_notes = np.sum(cs[cs >= 2] - 1)
    return {
        'max_chord': cs.max(), 'p99_chord': np.percentile(cs, 99),
        'multi_events': multi_events, 'multi_notes': multi_notes,
        'total': len(tc),
    }

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
for nm, p in [
    ('Melodiniq', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')),
    ('Verrückt(16.5)', os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')),
    ('夢降日(16.6)', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')),
    ('DerSchneid(17.5)', os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json')),
]:
    s = chord_stats(p)
    print(f'{nm:<18} 最大和弦={s["max_chord"]} P99={s["p99_chord"]:.0f} 多押事件={s["multi_events"]} 多押音符={s["multi_notes"]}/{s["total"]} ({s["multi_notes"]/max(s["total"],1)*100:.0f}%)')
print('DONE')