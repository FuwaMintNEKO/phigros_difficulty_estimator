# -*- coding: utf-8 -*-
"""验证: Melodiniq tap 24分 的同线/跨线分布"""
import os, sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes

p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
all_notes, jls, bpm_tl = collect_all_notes(cd)
times = np.array([n['time'] for n in all_notes])
types = np.array([n['type'] for n in all_notes])
jl_idx = np.array([n['judge_line_idx'] for n in all_notes])
bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])

tap = types == 1
ts = times[tap]; jls_ = jl_idx[tap]; bps = bpm_arr[tap]
o = np.argsort(ts); ts = ts[o]; jls_ = jls_[o]; bps = bps[o]
its = np.diff(ts)
same = jls_[1:] == jls_[:-1]
secs = its / 32.0 * 60.0 / np.maximum(bps[1:], 1.0)
thr24 = 60.0 / (bps[1:] * 6)
is24 = secs < thr24
print(f'tap 24分: {is24.sum()} 个')
print(f'  其中同线: {np.sum(is24 & same)} ({(np.sum(is24 & same)/max(is24.sum(),1))*100:.0f}%)')
print(f'  其中跨线: {np.sum(is24 & ~same)} ({(np.sum(is24 & ~same)/max(is24.sum(),1))*100:.0f}%)')
print(f'\nfast_ms_050 只用同线 → 只捕获 {np.sum(is24 & same)} 个')
print(f'→ 跨线 24分 ({np.sum(is24 & ~same)} 个) 被丢弃!')
print(f'\n对比 Verrückt: 是否也跨线?')
p2 = os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')
with open(p2, 'rb') as f:
    cd2, _ = load_chart_from_bytes(f.read())
all_notes2, jls2, bpm_tl2 = collect_all_notes(cd2)
times2 = np.array([n['time'] for n in all_notes2])
types2 = np.array([n['type'] for n in all_notes2])
jl_idx2 = np.array([n['judge_line_idx'] for n in all_notes2])
bpm_arr2 = np.array([n.get('bpm', 120.0) for n in all_notes2])
tap2 = types2 == 1
ts2 = times2[tap2]; jls2_ = jl_idx2[tap2]; bps2 = bpm_arr2[tap2]
o2 = np.argsort(ts2); ts2 = ts2[o2]; jls2_ = jls2_[o2]; bps2 = bps2[o2]
its2 = np.diff(ts2)
same2 = jls2_[1:] == jls2_[:-1]
secs2 = its2 / 32.0 * 60.0 / np.maximum(bps2[1:], 1.0)
thr242 = 60.0 / (bps2[1:] * 6)
is242 = secs2 < thr242
print(f'Verrückt tap 24分: {is242.sum()} 个 (同线={np.sum(is242 & same2)}, 跨线={np.sum(is242 & ~same2)})')
print('DONE')