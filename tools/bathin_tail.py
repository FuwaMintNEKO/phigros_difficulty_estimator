# -*- coding: utf-8 -*-
"""Bathin: cv差速事件分布 + 尾杀密度分析"""
import os, sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

p = os.path.join(_ROOT, 'data', 'phira', 'json', '7516.json')
with open(p, 'rb') as f:
    cd, raw = load_chart_from_bytes(f.read())
# 原始 PEC cv
with open(p, encoding='utf-8', errors='replace') as f:
    lines = f.read().splitlines()
cv = []
for ln in lines:
    ln = ln.strip()
    if ln.startswith('cv'):
        parts = ln.split()
        if len(parts) >= 4:
            cv.append((int(parts[1]), float(parts[2]), float(parts[3])))
cv = sorted(cv, key=lambda x: x[1])
vals = np.array([v for _, _, v in cv])
print(f'cv事件: {len(cv)}  值范围: {vals.min():.1f} ~ {vals.max():.1f}')
print(f'值>50: {np.sum(vals>50)}  >100: {np.sum(vals>100)}  ==125: {np.sum(vals==125)}')
print(f'值分布: P50={np.percentile(vals,50):.1f} P90={np.percentile(vals,90):.1f} P99={np.percentile(vals,99):.1f}')
# 相邻变化幅度
d = np.abs(np.diff(vals))
print(f'相邻变化: max={d.max():.1f} mean={d.mean():.2f} P90={np.percentile(d,90):.1f} 变化>10的次数: {np.sum(d>10)}')
# 长条与变速的交互: 长条期间是否有变速?
# 收集长条时间范围
n_all = []
for jl in cd.get('judgeLineList', []):
    n_all.extend(jl.get('notesAbove', []))
    n_all.extend(jl.get('notesBelow', []))
holds = [n for n in n_all if n.get('type') == 3]
print(f'\n长条数: {len(holds)}')
# 长条时间: time ~ time+holdTime, 统计重叠的变速事件
cv_times = set(round(t, 2) for _, t, _ in cv)
overlap = 0
for h in holds:
    t0, ht = h.get('time', 0), h.get('holdTime', 0)
    t1 = t0 + ht
    for _, t, v in cv:
        if t0 <= t <= t1 and v != 1.0:
            overlap += 1
            break
print(f'长条期间有变速(非1.0)的长条数: {overlap}/{len(holds)}')
# 尾杀: 最后20%时间的密度
times = np.array(sorted(n.get('time', 0) for n in n_all))
dur = times.max() / 1000.0  # ticks? 看单位
print(f'\n时间范围(ticks): {times.min():.0f} ~ {times.max():.0f}')
# 提取器单位: time是ticks
end = times.max()
tail = times[times >= end * 0.8]
head = times[times < end * 0.8]
print(f'尾段(最后20%): {len(tail)}音符 / 前段: {len(head)}音符')
print('DONE')