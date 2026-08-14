# -*- coding: utf-8 -*-
"""RPE type 语义权威确认: 通过大量RPE谱统计 endTime-startTime 区分瞬时/持续"""
import os, sys, io, json, numpy as np, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# 抽样统计: 每种type的 endTime-startTime 分布 (拍)
files = glob.glob(os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '*.json'))[:300]
from collections import defaultdict
dur_by_type = defaultdict(list)
for f in files:
    try:
        raw = json.load(open(f, encoding='utf-8'))
        if 'judgeLineList' not in raw: continue
        for jl in raw.get('judgeLineList', []):
            for n in jl.get('notes', []):
                st = n.get('startTime'); et = n.get('endTime')
                ty = n.get('type')
                if isinstance(st, list) and isinstance(et, list) and len(st)>=3 and len(et)>=3:
                    s = st[0]*4 + st[1]/max(st[2],1)
                    e = et[0]*4 + et[1]/max(et[2],1)
                    dur_by_type[ty].append(abs(e-s))
    except Exception:
        pass
print('RPE type 的 endTime-startTime(拍) 分布 (抽样300谱):')
for ty in sorted(dur_by_type):
    d = np.array(dur_by_type[ty])
    p50 = np.percentile(d, 50)
    p90 = np.percentile(d, 90)
    pct_zero = np.mean(d < 0.01) * 100
    print(f'  type{ty}: n={len(d)} P50={p50:.3f} P90={p90:.3f} 零持续={pct_zero:.0f}%')
print('\n零持续=瞬时音符(tap/drag/flick), 有持续=hold')
print('DONE')