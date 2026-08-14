# -*- coding: utf-8 -*-
"""スタートリップ vs 官方8级 vs 官方12级 三方特征对比"""
import sys, os
sys.path.insert(0, r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator')
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json

DL = r'C:\Users\NaNK\Downloads'
_ROOT = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator'

with open(os.path.join(DL, 'スタートリップ(12.2).json'), 'rb') as f:
    cd, pe = load_chart_from_bytes(f.read())
f_self = extract_features(cd)

song_diffs = load_difficulty_tsv(os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv'))
cfs = find_chart_files(os.path.join(_ROOT, 'data', 'chart'))

def collect(lo, hi):
    out = []
    for fn, info in cfs.items():
        sid = info['song_id']
        if sid not in song_diffs:
            continue
        diffs = song_diffs[sid]
        for lv in ['EZ', 'HD', 'IN', 'AT']:
            if lv in info['levels'] and lv in diffs and lo <= diffs[lv] <= hi:
                try:
                    f8 = extract_features(load_chart_json(info['levels'][lv]))
                    if f8:
                        out.append((fn, lv, diffs[lv], f8))
                except Exception:
                    pass
    return out

g8 = collect(7.5, 8.5)
g12 = collect(11.5, 12.5)
print(f'8级谱 n={len(g8)}, 12级谱 n={len(g12)}')

def avg(fs, k):
    vals = [f.get(k, 0) for _, _, _, f in fs]
    return sum(vals) / len(vals) if vals else float('nan')

KEYS = [
    'total_notes', 'duration_sec', 'notes_per_second', 'real_core_notes_per_second',
    'core_peak_density_1sec_top5avg', 'above_avg_density_mean', 'above_avg_duration_sec',
    'bpm', 'bpm_max', 'tempo_change_count',
    'hold_ratio', 'hold_notes_ratio', 'chord_ratio',
    'position_entropy', 'position_range_used', 'avg_chord_size_poly',
    'stair_density', 'stair_speed_avg', 'trill_density', 'jack_count', 'jack_max_run',
    'jline_movement_density', 'jline_rotate_density', 'jline_disappear_density',
    'speed_volatility', 'rhythm_entropy', 'above_below_cross',
    'multi_finger_3plus_events', 'pattern_switch_rate', 'direction_irregularity',
]
print(f'{"特征":<34} {"スタートリップ":>10} {"8级均值":>10} {"12级均值":>10}')
for k in KEYS:
    print(f'{k:<34} {f_self.get(k, 0):>10.3f} {avg(g8, k):>10.3f} {avg(g12, k):>10.3f}')
