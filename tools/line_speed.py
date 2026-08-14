# -*- coding: utf-8 -*-
"""Bathin 线间差速: 每条线的cv速度分布"""
import os, sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'json', '7516.json')
with open(p, encoding='utf-8', errors='replace') as f:
    lines = f.read().splitlines()
cv_by_line = {}
for ln in lines:
    ln = ln.strip()
    if ln.startswith('cv'):
        parts = ln.split()
        if len(parts) >= 4:
            li, t, v = int(parts[1]), float(parts[2]), float(parts[3])
            cv_by_line.setdefault(li, []).append(v)
print('Bathin 线数:', len(cv_by_line))
print(f'{"线":>4}{"事件数":>8}{"P50":>8}{"max":>8}')
all_med = []
for li in sorted(cv_by_line):
    v = np.array(cv_by_line[li])
    med = np.median(v)
    all_med.append(med)
    print(f'{li:>4}{len(v):>8}{med:>8.1f}{v.max():>8.1f}')
all_med = np.array(all_med)
print(f'\n线间P50差异: min={all_med.min():.1f} max={all_med.max():.1f} std={all_med.std():.1f}')
# 对比官谱: 抽查一个官谱的线间速度
print('\n对比官谱样例(非RPE, 标准JSON):')
import json, glob
cnt = 0
for f in glob.glob(os.path.join(_ROOT, 'data', 'phira', 'json', '*.json')):
    try:
        with open(f, encoding='utf-8', errors='replace') as fh:
            d = json.load(fh)
        if not isinstance(d, dict) or 'judgeLineList' not in d: continue
        if d.get('META', {}).get('RPEVersion'): continue
        meds = []
        for jl in d['judgeLineList']:
            vals = [ev.get('value', 1.0) for ev in jl.get('speedEvents', [])]
            if vals: meds.append(np.median(vals))
        if meds:
            meds = np.array(meds)
            print(f'  {os.path.basename(f)}: 线间P50 std={meds.std():.1f} range={meds.min():.1f}-{meds.max():.1f}')
            cnt += 1
        if cnt >= 5: break
    except Exception:
        pass
print('DONE')