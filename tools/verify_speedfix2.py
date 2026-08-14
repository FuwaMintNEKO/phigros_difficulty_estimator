# -*- coding: utf-8 -*-
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
p1 = os.path.join(DL, '夢の降る日に', '5333883479687925.json')
p2 = os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')
with open(p1, 'rb') as f: cd1, _ = load_chart_from_bytes(f.read())
with open(p2, 'rb') as f: cd2, _ = load_chart_from_bytes(f.read())
f1 = extract_features(cd1, speed=1.0); f2 = extract_features(cd2, speed=1.0)
for k in ['speed_mean','speed_std','speed_max','speed_volatility']:
    print(f'{k}: 高仿={f1.get(k,0):.2f} 官谱={f2.get(k,0):.2f}')
print('DONE')