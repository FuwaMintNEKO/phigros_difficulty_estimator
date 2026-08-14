# -*- coding: utf-8 -*-
"""kyou标签 vs 我们13维度标签 对照验证
"""
import os, sys, pickle, numpy as np, io, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
kyou = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'kyou_tags.json'), encoding='utf-8'))
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = cache['official']
TH = json.load(open(os.path.join(_ROOT, 'data', 'tag_thresholds.json'), encoding='utf-8'))
DIM = [('底力', 'above_avg_density_mean'), ('多押', 'weighted_mf_score_per_sec'),
       ('楼梯', 'stair_speed_avg'), ('32分', 'thirtysecond_run_ratio'),
       ('爆发', 'fast_ms_100_ratio'), ('读谱', 'jline_movement_density'),
       ('变速', 'tempo_change_log_density'), ('耐力', 'above_avg_duration_sec'),
       ('高BPM', 'bpm'), ('纵连', 'jack_density'), ('叠键', 'chord_jack_3plus_pairs'),
       ('位移', 'movement_per_second')]
def our_tags(f):
    out = []
    for name, fk in DIM:
        if f.get(fk, 0) >= TH.get(name, 1e9): out.append(name)
    if f.get('tracks_6plus_sec', 0) / max(f.get('tracks_active_sec', 1), 0.01) >= TH.get('定轨', 1): out.append('定轨')
    return out
def norm(s):
    return re.sub(r'[^a-z0-9\u4e00-\u9fff]', '', s.lower())

rows = []
for k in kyou:
    kn = norm(k['song'])
    for r in official:
        if r['level'] in ('IN', 'AT') and kn and kn in norm(r['name']):
            rows.append({'kyou': k['tag'], 'song': k['song'], 'our': our_tags(r['feats']), 'diff': r['diff']})
            break
print(f'匹配: {len(rows)} / {len(kyou)}')
# 一致性: kyou主类 vs 我们特征
def classify_kyou(t):
    return t
def classify_our(ts):
    if len(ts) >= 4: return '综合'
    if '底力' in ts or '耐力' in ts: return '硬抗'
    if '读谱' in ts: return '读谱'
    if '多押' in ts or '位移' in ts: return '多指'
    if '楼梯' in ts or '32分' in ts or '纵连' in ts or '叠键' in ts: return '拆谱'
    if 'jline' in ts: return '定位'
    return '无'
# jline 单独: 定位判定
from collections import Counter, defaultdict
agree = Counter(); total = Counter(); detail = defaultdict(list)
for r in rows:
    kt = r['kyou']; ot = classify_our(r['our'])
    total[kt] += 1
    if kt == ot: agree[kt] += 1
    detail[kt].append((r['song'], ot, '+'.join(r['our']) if r['our'] else '-'))
print('\n一致性 (kyou主类 vs 我们特征分类):')
for kt in ['硬抗', '综合', '定位', '读谱', '拆谱', '多指']:
    if total[kt]:
        print(f'  {kt}: n={total[kt]} 一致={agree[kt]} ({100*agree[kt]/total[kt]:.0f}%)')
# 盲区: kyou有标签但我们无标签
print('\n盲区 (kyou有标签, 我们0特征标签):')
for kt in ['硬抗', '综合', '定位', '读谱', '拆谱', '多指']:
    blind = [(s, o) for s, o, t in detail[kt] if o == '无']
    if blind:
        print(f'  [{kt}] {len(blind)}首: {[b[0] for b in blind[:6]]}')
print('DONE')
