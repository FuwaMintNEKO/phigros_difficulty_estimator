# -*- coding: utf-8 -*-
"""type4 视觉特征: 是否像drag(长条滑动) 还是像tap"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
raw = json.load(open(p, encoding='utf-8'))
notes_by_type = {}
for jl in raw.get('judgeLineList', []):
    for n in jl.get('notes', []):
        ty = n.get('type')
        notes_by_type.setdefault(ty, []).append(n)
for ty in [1, 4, 2, 3]:
    ns = notes_by_type.get(ty, [])
    if not ns: continue
    # 检查字段差异
    has_end = sum(1 for n in ns if n.get('endTime') is not None)
    speeds = [n.get('speed', 1.0) for n in ns]
    vts = [n.get('visibleTime', 999999) for n in ns]
    pos = [n.get('positionX', 0) for n in ns]
    print(f'\ntype{ty} (n={len(ns)}):')
    print(f'  带endTime: {has_end}/{len(ns)}')
    print(f'  speed: min={min(speeds):.2f} max={max(speeds):.2f}')
    print(f'  visibleTime: min={min(vts):.0f} (999999=默认)')
    print(f'  positionX: min={min(pos):.0f} max={max(pos):.0f}')
    # endTime 与 startTime 的差值 (如果是hold应该有差)
    if has_end:
        ds = []
        for n in ns:
            st = n.get('startTime'); et = n.get('endTime')
            if isinstance(st, list) and isinstance(et, list) and len(st)>=3 and len(et)>=3:
                ds.append(abs(st[0]*4+st[1]/max(st[2],1) - (et[0]*4+et[1]/max(et[2],1))))
        if ds:
            print(f'  endTime-startTime(拍): P50={np.median(ds):.3f} max={max(ds):.3f}')
print('DONE')