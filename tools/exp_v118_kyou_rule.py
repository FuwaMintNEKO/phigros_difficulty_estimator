# -*- coding: utf-8 -*-
"""kyou 6类分类规则验证 (特征规则 → kyou类)
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

def classify(f):
    dens = f.get('above_avg_density_mean', 0); eff = f.get('eff_avg_tps_1s', 0)
    wmf = f.get('weighted_mf_score_per_sec', 0); jline = f.get('jline_movement_density', 0)
    dura = f.get('above_avg_duration_sec', 0); stair = f.get('stair_speed_avg', 0)
    ts = f.get('thirtysecond_run_ratio', 0); ch8 = f.get('chord_events_peak_8s', 0)
    score = {}
    score['多指'] = 1.0 if wmf >= 10 and ch8 >= 16 else 0.0
    score['硬抗'] = 1.0 if dens >= 8.0 or (dens >= 7.0 and dura >= 500) else 0.0
    score['读谱'] = 1.0 if dens <= 7.5 and jline >= 120 else 0.0
    score['拆谱'] = 1.0 if stair >= 13.0 and wmf < 8 else 0.0
    mx = max(score.values())
    if mx > 0:
        cands = [k for k, v in score.items() if v == mx]
        return cands[0]
    return '综合'

from collections import Counter
total = Counter(); correct = Counter(); conf = {}
for cat, f in rows:
    pred = classify(f)
    total[cat] += 1
    if pred == cat: correct[cat] += 1
    conf.setdefault(cat, Counter())[pred] += 1
print('分类准确率 (特征规则 vs kyou共识):')
for cat in ['硬抗', '综合', '定位', '读谱', '拆谱', '多指']:
    if total[cat]:
        print(f'  {cat}: n={total[cat]} 正确={correct[cat]} ({100*correct[cat]/total[cat]:.0f}%) 分布={dict(conf[cat].most_common(3))}')
allc = sum(total.values()); allr = sum(correct.values())
print(f'\n总: {allr}/{allc} = {100*allr/allc:.0f}%')
print('DONE')
