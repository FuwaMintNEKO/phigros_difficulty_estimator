# -*- coding: utf-8 -*-
"""验证: speed统一裁剪后 高仿vs官谱"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
pairs = [
    ('夢降日', os.path.join(DL, '夢の降る日に', '5333883479687925.json'),
              os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')),
    ('DerSchneid', os.path.join(DL, 'Der Schneid(1)', '1903581575578621.json'),
              os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json')),
]
for nm, p1, p2 in pairs:
    with open(p1, 'rb') as f: cd1, _ = load_chart_from_bytes(f.read())
    with open(p2, 'rb') as f: cd2, _ = load_chart_from_bytes(f.read())
    f1 = extract_features(cd1, speed=1.0); f2 = extract_features(cd2, speed=1.0)
    print(f'\n{nm}:')
    for k in ['speed_mean','speed_std','speed_max','speed_volatility','speed_change_total_impact']:
        print(f'  {k:<26} 高仿={f1.get(k,0):.2f} 官谱={f2.get(k,0):.2f}')
print('DONE')