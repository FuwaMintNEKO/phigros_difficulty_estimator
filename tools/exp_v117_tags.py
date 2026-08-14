# -*- coding: utf-8 -*-
"""官谱15+ 难点标签 v3: +纵连/叠键/位移/爆发峰值 维度
"""
import os, sys, pickle, numpy as np, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = cache['official']
sel = [r for r in official if r['diff'] >= 15]
DIM = [
    ('底力', 'above_avg_density_mean', 75), ('多押', 'weighted_mf_score_per_sec', 75),
    ('楼梯', 'stair_speed_avg', 75), ('高速', 'fast_ms_100_ratio', 75),
    ('爆发', 'fast_ms_050_ratio', 75), ('读谱', 'jline_movement_density', 75),
    ('变速', 'tempo_change_log_density', 75), ('耐力', 'above_avg_duration_sec', 75),
    ('高BPM', 'bpm', 75), ('纵连', 'jack_density', 75), ('叠键', 'chord_jack_3plus_pairs', 75),
    ('位移', 'movement_per_second', 75),
]
pcts = {}
for name, fk, q in DIM:
    vals = [r['feats'].get(fk, 0) for r in sel]
    pcts[name] = float(np.percentile(vals, q))
print('阈值:', {k: round(v, 2) for k, v in pcts.items()})

def labels(r):
    f = r['feats']
    out = []
    for name, fk, q in DIM:
        if f.get(fk, 0) >= pcts[name]: out.append(name)
    t6 = f.get('tracks_6plus_sec', 0) / max(f.get('tracks_active_sec', 1), 0.01)
    if t6 >= 0.02: out.append('定轨')
    return out

# 玩家关注谱验证
focus = ['csqn', 'Palescreen', 'KMoeVIP', '祈', 'CROSSSOUL', 'KIZUNA', 'DerSchneid', '夢の降る日に', 'Verruckt', 'StardustRAY', 'ReEndofaDream', 'Rrharil', 'DistortedFate', 'GOODRAGE', '白と黒']
print('\n===== 玩家关注谱标签 =====')
for r in official:
    if r['diff'] >= 15 and any(kw.lower() in r['name'].lower() for kw in focus):
        print(f'{r["diff"]:.1f} {r["level"]:<3} {r["name"][:24]:<26} {"+".join(labels(r)) if labels(r) else "-"}')
print('DONE')
