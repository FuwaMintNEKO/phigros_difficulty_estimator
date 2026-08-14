# -*- coding: utf-8 -*-
"""官谱(标准JSON) speedEvents 值分布 vs Bathin PEC cv"""
import os, sys, io, json, glob, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes

# 遍历官谱json (有judgeLineList且无RPEVersion)
files = glob.glob(os.path.join(_ROOT, 'data', 'phira', 'json', '*.json'))
all_vals = []
examples = []
for f in files[:400]:
    try:
        with open(f, 'rb') as fh:
            cd, raw = load_chart_from_bytes(fh.read())
        if not isinstance(cd, dict) or 'judgeLineList' not in cd:
            continue
        meta = cd.get('META', {})
        if meta.get('RPEVersion'):
            continue  # RPE谱跳过
        jls = cd.get('judgeLineList', [])
        vals = []
        for jl in jls:
            for ev in jl.get('speedEvents', []):
                if 'value' in ev:
                    vals.append(ev['value'])
        if vals:
            all_vals.extend(vals)
            examples.append((os.path.basename(f), max(vals)))
    except Exception:
        pass
av = np.array(all_vals)
print(f'官谱(标准) speedEvents 值: n={len(av)}')
print(f'  P50={np.percentile(av,50):.2f} P90={np.percentile(av,90):.2f} P99={np.percentile(av,99):.2f} max={av.max():.1f}')
print(f'  值>50: {np.sum(av>50)}  >100: {np.sum(av>100)}  >1000: {np.sum(av>1000)}')
examples.sort(key=lambda x: -x[1])
print('  最大值的谱:', examples[:5])
print('\nBathin PEC cv: P50=13.0 P90=17.0 max=125')
print('DONE')