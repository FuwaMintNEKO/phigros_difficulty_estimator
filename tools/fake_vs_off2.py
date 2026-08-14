# -*- coding: utf-8 -*-
"""修复后高仿对照复测"""
import os, sys, io, numpy as np
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
    with open(p1, 'rb') as f:
        cd1, _ = load_chart_from_bytes(f.read())
    with open(p2, 'rb') as f:
        cd2, _ = load_chart_from_bytes(f.read())
    f1 = extract_features(cd1, speed=1.0)
    f2 = extract_features(cd2, speed=1.0)
    keys = sorted(set(f1.keys()) | set(f2.keys()))
    diffs = []
    for k in keys:
        v1 = f1.get(k, 0); v2 = f2.get(k, 0)
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if abs(v1 - v2) > max(abs(v1), abs(v2), 1.0) * 0.05:
                diffs.append((k, v1, v2))
    print(f'\n{nm} 显著差异特征 ({len(diffs)}):')
    for k, v1, v2 in sorted(diffs, key=lambda x: -abs(x[1]-x[2]))[:20]:
        print(f'  {k:<36} 高仿={v1:>10.2f} 官谱={v2:>10.2f}')
print('DONE')