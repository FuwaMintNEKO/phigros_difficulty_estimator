# -*- coding: utf-8 -*-
"""新增特征: tap_24分绝对计数 + 密度 (修复占比被长谱稀释)"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes

# 计算候选特征值 (Melodiniq vs 官谱16.5+)
def tap_burst_features(path):
    with open(path, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    all_notes, jls, bpm_tl = collect_all_notes(cd)
    times = np.array([n['time'] for n in all_notes])
    types = np.array([n['type'] for n in all_notes])
    bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
    dur = times.max() / 32.0 * 60.0 / max(bpm_arr.max(), 1.0)  # 粗估时长
    # 用真实时长
    from feature_extractor import time_to_seconds
    tsec = np.array([time_to_seconds(t, max(b,1.0), bpm_tl) for t, b in zip(times, bpm_arr)])
    dur = tsec.max()
    tap = types == 1
    ts = times[tap]; bps = bpm_arr[tap]
    o = np.argsort(ts); ts = ts[o]; bps = bps[o]
    its = np.diff(ts)
    secs = its / 32.0 * 60.0 / np.maximum(bps[1:], 1.0)
    thr24 = 60.0 / (bps[1:] * 6)
    thr16 = 60.0 / (bps[1:] * 4)
    n24 = np.sum(secs < thr24)
    n16 = np.sum((secs >= thr24) & (secs < thr16))
    return {
        'tap24_count': n24, 'tap16_count': n16,
        'tap24_density': n24 / dur, 'tap16_density': n16 / dur,
        'tap24_16_total': n24 + n16,
        'dur': dur,
    }

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
cases = [
    ('Melodiniq', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')),
    ('Verrückt(16.5)', os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')),
    ('夢降日(16.6)', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')),
    ('DerSchneid(17.5)', os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json')),
]
print(f'{"谱":<16}{"时长":>6}{"tap24":>7}{"tap16":>7}{"24+16":>7}{"24密度":>8}{"16密度":>8}')
for nm, p in cases:
    s = tap_burst_features(p)
    print(f'{nm:<16}{s["dur"]:>6.0f}{s["tap24_count"]:>7}{s["tap16_count"]:>7}{s["tap24_16_total"]:>7}{s["tap24_density"]:>8.3f}{s["tap16_density"]:>8.3f}')
print('DONE')