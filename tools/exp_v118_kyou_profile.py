# -*- coding: utf-8 -*-
"""kyou分类的特征画像: 硬抗/定位/读谱/拆谱/多指 的典型特征值
"""
import os, sys, pickle, numpy as np, io, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
kyou = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'kyou_tags.json'), encoding='utf-8'))
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = cache['official']
def norm(s):
    return re.sub(r'[^a-z0-9\u4e00-\u9fff]', '', s.lower())
rows = []
for k in kyou:
    kn = norm(k['song'])
    for r in official:
        if r['level'] in ('IN', 'AT') and kn and kn in norm(r['name']):
            rows.append((k['tag'], r['feats']))
            break
KEYS = ['above_avg_density_mean', 'eff_avg_tps_1s', 'weighted_mf_score_per_sec', 'stair_speed_avg',
        'thirtysecond_run_ratio', 'jline_movement_density', 'above_avg_duration_sec', 'avg_movement',
        'position_iqr', 'position_entropy', 'movement_per_second', 'bpm', 'chord_events_peak_8s', 'drag_ratio']
print(f'{"分类":<6}{"n":>4} | ' + ' '.join(f'{k[:6]:>7}' for k in KEYS))
for cat in ['硬抗', '综合', '定位', '读谱', '拆谱', '多指']:
    fs = [f for c, f in rows if c == cat]
    if not fs: continue
    vals = []
    for k in KEYS:
        vs = [f.get(k, 0) for f in fs]
        vals.append(f'{np.median(vs):>7.2f}')
    print(f'{cat:<6}{len(fs):>4} | ' + ' '.join(vals))
print('DONE')
