# -*- coding: utf-8 -*-
"""颜(RPE) 音符speed原始值 + Aurora(PEC) cv命令"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# 颜 (RPE)
p = os.path.join(_ROOT, 'data', 'phira', 'json', '37193.json')
with open(p, encoding='utf-8') as f:
    d = json.load(f)
print('颜 keys:', list(d.keys()))
jls = d.get('judgeLineList', [])
alln = []
for jl in jls:
    alln.extend(jl.get('notesAbove', []))
    alln.extend(jl.get('notesBelow', []))
print('颜 音符数:', len(alln))
if alln:
    print('音符keys:', list(alln[0].keys()))
    sp = [float(n.get('speed', 1.0)) for n in alln]
    import numpy as np
    sp = np.array(sp)
    print('speed: min={} P50={} P90={} max={}'.format(sp.min(), np.percentile(sp,50), np.percentile(sp,90), sp.max()))
    non1 = sp[sp != 1.0]
    print('非1.0 speed 数量:', len(non1), ' 值样例:', sorted(set(non1))[:20])
print()
# Aurora (PEC) cv命令
p2 = os.path.join(_ROOT, 'data', 'phira', 'json', '10203.json')
with open(p2, encoding='utf-8', errors='replace') as f:
    lines = f.read().splitlines()
cv_vals = []
for ln in lines:
    ln = ln.strip()
    if ln.startswith('cv'):
        parts = ln.split()
        if len(parts) >= 4:
            cv_vals.append(float(parts[3]))
print('Aurora cv数:', len(cv_vals), '值范围:', min(cv_vals) if cv_vals else '-', '-', max(cv_vals) if cv_vals else '-')
if cv_vals:
    import numpy as np
    v = np.array(cv_vals)
    print('  P50={} P90={}'.format(np.percentile(v,50), np.percentile(v,90)))
    print('  大值样例:', sorted(set(v))[-10:])
print('DONE')