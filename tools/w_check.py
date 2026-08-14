# -*- coding: utf-8 -*-
"""确认模型pkl内MANUAL_FLAT与boost_config是否一致"""
import os, sys, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT as BC
with open(os.path.join(_ROOT, 'models', '6dim_model_v11_10.pkl'), 'rb') as f:
    m = pickle.load(f)
MP = m.get('MANUAL_FLAT')
print('boost_config len:', len(BC), ' pkl len:', len(MP))
same = all(a == b for a, b in zip(BC, MP))
print('完全一致:', same)
if not same:
    for a, b in zip(BC, MP):
        if a != b:
            print('  差异:', a, 'vs', b)
# 相关权重当前值
for fname in ['drag_per_sec', 'density_transition_std', 'jack_max_run', 'eff_peak_tps_1s', 'above_avg_duration_sec', 'jline_movement_density']:
    for f, bl, co in MP:
        if f == fname:
            print(f'{fname}: co={co}')
print('DONE')