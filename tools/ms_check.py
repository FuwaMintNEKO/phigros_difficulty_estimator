# -*- coding: utf-8 -*-
"""验证: 更细毫秒窗口(<20ms/<15ms) 能否区分 Melodiniq vs Verrückt + 全量分布"""
import os, sys, io, json, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

# 重新算两个谱的毫秒间隔分布
def ms_stats(path):
    with open(path, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    # 从cd拿音符
    notes = []
    for jl in cd.get('judgeLineList', []):
        for n in jl.get('notesAbove', []):
            notes.append((n['time'], n.get('positionX', 0), 1))
        for n in jl.get('notesBelow', []):
            notes.append((n['time'], n.get('positionX', 0), 2))
    # RPE有notes? Melodiniq是RPE, 转换后有notesAbove
    if not notes:
        return None
    notes.sort()
    times = np.array([t for t, _, _ in notes])
    bpm = 193.0
    # ticks→秒: 1拍=32ticks, 秒=ticks/32*60/bpm
    # 但BPM多变, 粗略用平均
    intervals_sec = np.diff(times) / 32.0 * 60.0 / bpm
    # 同线
    lines = np.array([l for _, _, l in notes])
    same = lines[1:] == lines[:-1]
    its = intervals_sec[same]
    return {
        'n': len(notes),
        '<50ms': np.mean(its < 0.05),
        '<30ms': np.mean(its < 0.03),
        '<20ms': np.mean(its < 0.02),
        '<15ms': np.mean(its < 0.015),
        '<10ms': np.mean(its < 0.01),
    }

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
r1 = ms_stats(os.path.join(DL, '夢の降る日に', '5333883479687925.json'))
r2 = ms_stats(os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json'))
r3 = ms_stats(os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json'))
r4 = ms_stats(os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json'))
print(f'{"谱":<20}{"<50ms":>8}{"<30ms":>8}{"<20ms":>8}{"<15ms":>8}{"<10ms":>8}')
for nm, r in [('夢降日高仿', r1), ('夢降日官谱', r2), ('Melodiniq', r3), ('Verrückt', r4)]:
    if r:
        print(f'{nm:<20}{r["<50ms"]:>8.3f}{r["<30ms"]:>8.3f}{r["<20ms"]:>8.3f}{r["<15ms"]:>8.3f}{r["<10ms"]:>8.3f}')
print('DONE')