# -*- coding: utf-8 -*-
"""社区谱难点标签: ranked + 未上架高评分 (用官谱阈值)
输出: data/phira/ranked_tags.csv + unranked_tags.csv
"""
import os, sys, pickle, numpy as np, io, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = cache['official']
sel = [r for r in official if r['diff'] >= 15]
DIM = [
    ('底力', 'above_avg_density_mean', 75), ('多押', 'weighted_mf_score_per_sec', 75),
    ('楼梯', 'stair_speed_avg', 75), ('32分', 'thirtysecond_run_ratio', 75),
    ('爆发', 'fast_ms_100_ratio', 75), ('读谱', 'jline_movement_density', 75),
    ('变速', 'tempo_change_log_density', 75), ('耐力', 'above_avg_duration_sec', 75),
    ('高BPM', 'bpm', 75), ('纵连', 'jack_density', 75), ('叠键', 'chord_jack_3plus_pairs', 75),
    ('位移', 'movement_per_second', 75), ('定轨', None, 90),
]
pcts = {}
for name, fk, q in DIM:
    if fk is None:
        vals = [r['feats'].get('tracks_6plus_sec', 0) / max(r['feats'].get('tracks_active_sec', 1), 0.01) for r in sel]
    else:
        vals = [r['feats'].get(fk, 0) for r in sel]
    pcts[name] = float(np.percentile(vals, q))
# 纵连/叠键: p75=0 会导致所有>0都触发 — 用p90
for name in ['纵连', '叠键']:
    fk = 'jack_density' if name == '纵连' else 'chord_jack_3plus_pairs'
    vals = [r['feats'].get(fk, 0) for r in sel]
    pcts[name] = float(np.percentile(vals, 90))

def labels(f):
    out = []
    for name, fk, q in DIM:
        if fk is None:
            v = f.get('tracks_6plus_sec', 0) / max(f.get('tracks_active_sec', 1), 0.01)
        else:
            v = f.get(fk, 0)
        if v >= pcts[name]: out.append(name)
    return '+'.join(out) if out else '-'

# ranked
ranked = cache['ranked']
rows = []
for r in ranked:
    if not r['diff'] or r['diff'] <= 10: continue
    rows.append({'id': r['id'], 'name': r['name'], 'level': r['level'], 'diff': r['diff'], 'tags': labels(r['feats'])})
with open(os.path.join(_ROOT, 'data', 'phira', 'ranked_tags.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'name', 'level', 'diff', 'tags'])
    for x in rows: w.writerow([x['id'], x['name'], x['level'], x['diff'], x['tags']])
print(f'ranked 标签: {len(rows)} 张 → ranked_tags.csv')

# 16+ 高评分展示
high = [x for x in rows if x['diff'] >= 16 and x['tags'] != '-']
high.sort(key=lambda x: -x['diff'])
print(f'\n===== 上架 16+ 有标签谱 (top30) =====')
for x in high[:30]:
    print(f'{x["diff"]:.1f} {x["level"][:8]:<10} {x["name"][:24]:<26} {x["tags"]}')
print('DONE')
