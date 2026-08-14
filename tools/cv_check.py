# -*- coding: utf-8 -*-
"""Bathin cv事件值分布 + speedEvents合成后特征"""
import os, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
p = os.path.join(_ROOT, 'data', 'phira', 'json', '7516.json')
with open(p, encoding='utf-8', errors='replace') as f:
    lines = f.read().splitlines()
cv_vals = []
for ln in lines:
    ln = ln.strip()
    if ln.startswith('cv'):
        parts = ln.split()
        if len(parts) >= 4:
            cv_vals.append((float(parts[1]), float(parts[2]), float(parts[3])))
print('cv事件:', len(cv_vals))
vals = sorted(set(v for _, _, v in cv_vals))
print('cv速度值集合:', vals)
# 时间范围
ts = [t for _, t, _ in cv_vals]
print('cv时间范围:', min(ts), '-', max(ts))
# speedEvents 合成检查
from unified_parser import load_chart_from_bytes
with open(p, 'rb') as f:
    cd, raw = load_chart_from_bytes(f.read())
# cd 是标准格式 judge_lines
print('\ncd类型keys:', list(cd.keys())[:10] if isinstance(cd, dict) else 'list')
if isinstance(cd, dict):
    jls = cd.get('judgeLineList', [])
    se_total = 0
    for jl in jls:
        se_total += len(jl.get('speedEvents', []))
    print('合成speedEvents总数:', se_total)
    # 看第一条线的speedEvents
    for jl in jls:
        ses = jl.get('speedEvents', [])
        if ses:
            print('样例speedEvents:', ses[:3])
            break
else:
    print('cd是list, len:', len(cd))
    se_total = 0
    for jl in cd:
        se_total += len(jl.get('speedEvents', []))
    print('合成speedEvents总数:', se_total)
print('DONE')