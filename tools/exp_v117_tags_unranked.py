# -*- coding: utf-8 -*-
"""未上架5894张 难点标签 全量生成 → data/phira/unranked_tags.csv
"""
import os, sys, io, json, csv, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
TH = json.load(open(os.path.join(_ROOT, 'data', 'tag_thresholds.json'), encoding='utf-8'))
DIM = [('底力', 'above_avg_density_mean'), ('多押', 'weighted_mf_score_per_sec'),
       ('楼梯', 'stair_speed_avg'), ('32分', 'thirtysecond_run_ratio'),
       ('爆发', 'fast_ms_100_ratio'), ('读谱', 'jline_movement_density'),
       ('变速', 'tempo_change_log_density'), ('耐力', 'above_avg_duration_sec'),
       ('高BPM', 'bpm'), ('纵连', 'jack_density'), ('叠键', 'chord_jack_3plus_pairs'),
       ('位移', 'movement_per_second')]
def tags(f):
    out = []
    for name, fk in DIM:
        if f.get(fk, 0) >= TH.get(name, 1e9): out.append(name)
    if f.get('tracks_6plus_sec', 0) / max(f.get('tracks_active_sec', 1), 0.01) >= TH.get('定轨', 1): out.append('定轨')
    return '+'.join(out) if out else '-'

meta = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'unranked_all.json'), encoding='utf-8'))
meta_by_id = {c['id']: c for c in meta}
JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star')
files = sorted(f for f in os.listdir(JSON_DIR) if f.endswith('.json'))
rows = []
ok = fail = 0
import time
t0 = time.time()
for i, fn in enumerate(files):
    cid = int(fn[:-5])
    try:
        with open(os.path.join(JSON_DIR, fn), 'rb') as f:
            cd, raw = load_chart_from_bytes(f.read())
        feats = extract_features(cd, speed=1.0)
        if not feats:
            fail += 1; continue
        c = meta_by_id.get(cid, {})
        rows.append([cid, c.get('name', ''), c.get('level', ''), c.get('difficulty'), tags(feats)])
        ok += 1
    except Exception:
        fail += 1
    if (i + 1) % 1000 == 0:
        print(f'  {i+1}/{len(files)} 成功{ok} 失败{fail} {time.time()-t0:.0f}s', flush=True)
out_csv = os.path.join(_ROOT, 'data', 'phira', 'unranked_tags.csv')
with open(out_csv, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'name', 'level', 'difficulty', 'tags'])
    for r_ in rows: w.writerow(r_)
print(f'完成: {ok} 张 → {out_csv}')
print('DONE')
