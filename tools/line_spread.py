# -*- coding: utf-8 -*-
"""线间差速特征验证"""
import os, sys, io, numpy as np, glob, json, pickle, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]

def line_speed_stats(path):
    try:
        with open(path, 'rb') as f:
            cd, raw = load_chart_from_bytes(f.read())
        if not isinstance(cd, dict) or 'judgeLineList' not in cd:
            return None
        meds = []
        for jl in cd['judgeLineList']:
            evs = jl.get('speedEvents', [])
            if not evs: continue
            vals = []
            for ev in evs:
                if 'value' in ev: vals.append(ev['value'])
                elif 'start' in ev: vals.append(ev['start'] / 5.0)
            if vals: meds.append(np.median(vals))
        if len(meds) < 2: return None
        meds = np.array(meds)
        return np.std(meds), meds.max() - meds.min(), meds.min(), meds.max()
    except Exception:
        return None

ids = [r['id'] for r in ranked]
random.seed(1)
sample_ids = set(ids[i] for i in random.sample(range(len(ranked)), 200))
for r in ranked:
    if r['id'] in (7516, 47264, 15875, 59064, 37193, 10203):
        sample_ids.add(r['id'])

results = []
for r in ranked:
    if r['id'] not in sample_ids: continue
    p = os.path.join(_ROOT, 'data', 'phira', 'json', f"{r['id']}.json")
    if not os.path.exists(p): continue
    st = line_speed_stats(p)
    if st:
        results.append((r['name'][:20], r['id'], round(r['diff'],1), st[0], st[1], st[2], st[3]))
print(f'统计 {len(results)} 首的线间速度std:')
stds = np.array([x[3] for x in results])
print(f'  P25={np.percentile(stds,25):.2f} P50={np.percentile(stds,50):.2f} P75={np.percentile(stds,75):.2f} P90={np.percentile(stds,90):.2f} P95={np.percentile(stds,95):.2f} max={stds.max():.2f}')
print('\n线间std 最高15首:')
for x in sorted(results, key=lambda z: -z[3])[:15]:
    print(f'  {x[0]:<22} id={x[1]} diff={x[2]:.1f} line_std={x[3]:.2f} range={x[4]:.1f} ({x[5]:.1f}-{x[6]:.1f})')
print('\n锚点谱:')
for x in results:
    if x[1] in (7516, 47264, 15875, 59064):
        print(f'  {x[0]:<22} id={x[1]} diff={x[2]:.1f} line_std={x[3]:.2f} range={x[4]:.1f}')
print('DONE')