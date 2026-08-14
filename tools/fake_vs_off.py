# -*- coding: utf-8 -*-
"""官谱高仿对照: 配置一模一样的谱面特征应一致 (黄金标准测试)"""
import os, sys, io, json, numpy as np
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

for nm, p_fake, p_off in pairs:
    with open(p_fake, 'rb') as f:
        cd1, _ = load_chart_from_bytes(f.read())
    with open(p_off, 'rb') as f:
        cd2, _ = load_chart_from_bytes(f.read())
    f1 = extract_features(cd1, speed=1.0)
    f2 = extract_features(cd2, speed=1.0)
    print(f'\n===== {nm} 高仿 vs 官谱 =====')
    keys = sorted(set(f1.keys()) | set(f2.keys()))
    diffs = []
    for k in keys:
        v1 = f1.get(k, 0); v2 = f2.get(k, 0)
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if abs(v1 - v2) > max(abs(v1), abs(v2), 1.0) * 0.05:  # 5%以上差异
                diffs.append((k, v1, v2))
    print(f'显著差异特征 ({len(diffs)}):')
    for k, v1, v2 in sorted(diffs, key=lambda x: -abs(x[1]-x[2]))[:40]:
        print(f'  {k:<38} 高仿={v1:>12.2f} 官谱={v2:>12.2f} 差={v1-v2:+.2f}')
print('DONE')