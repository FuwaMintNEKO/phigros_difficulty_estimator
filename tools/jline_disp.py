# -*- coding: utf-8 -*-
"""jline特征重构: 用位移总量(两种格式可比) 替代事件计数"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes

def jline_disp(path):
    """判定线位移总量/秒 (官谱: moveEvents的|x位移|累计; RPE: moveXEvents的|start-end|累计/1350)"""
    with open(path, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    jls = cd.get('judgeLineList', [])
    total_disp = 0.0
    n_moves = 0
    for jl in jls:
        # 官谱: 顶层 moveEvents
        for ev in jl.get('judgeLineMoveEvents', []):
            total_disp += abs(ev.get('end', 0) - ev.get('start', 0))
            total_disp += abs(ev.get('end2', 0) - ev.get('start2', 0))
            n_moves += 1
        # RPE: eventLayers moveX/moveY (像素, /1350 /900 归一化)
        for layer in jl.get('eventLayers', []) or []:
            if not layer: continue
            for ev in layer.get('moveXEvents', []):
                total_disp += abs(ev.get('end', 0) - ev.get('start', 0)) / 1350.0
                n_moves += 1
            for ev in layer.get('moveYEvents', []):
                total_disp += abs(ev.get('end', 0) - ev.get('start', 0)) / 900.0
                n_moves += 1
    # 时长
    all_notes, _, bpm_tl = collect_all_notes(cd)
    times = np.array([n['time'] for n in all_notes])
    bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
    dur = times.max() / 32.0 * 60.0 / max(bpm_arr.max(), 1.0)
    return total_disp, n_moves, dur

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
for nm, p in [
    ('夢降日高仿', os.path.join(DL, '夢の降る日に', '5333883479687925.json')),
    ('夢降日官谱', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')),
    ('DerSchneid高仿', os.path.join(DL, 'Der Schneid(1)', '1903581575578621.json')),
    ('DerSchneid官谱', os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json')),
    ('Melodiniq', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')),
    ('Verrückt', os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')),
]:
    td, nm_, dur = jline_disp(p)
    print(f'{nm:<18} 位移总量={td:.1f} 位移/秒={td/dur:.2f} 事件={nm_} 时长={dur:.0f}s')
print('DONE')