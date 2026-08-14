# -*- coding: utf-8 -*-
"""实验B: 定位特征开发 — kyou定位(56首) vs 其他 的特征区分
找: 定位=位置精确击打, 需要哪些新特征?
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
            rows.append((k['tag'].replace('?', '').strip(), r['feats']))
            break
# 全特征对比 定位 vs 硬抗 vs 读谱
KEYS = ['avg_movement', 'max_movement', 'position_iqr', 'position_std', 'position_entropy', 'position_range_used',
        'cross_hand_density', 'lane_switch_density', 'crossline_chain_max', 'jline_relative_cross',
        'short_interval_ratio', 'very_short_interval_ratio', 'avg_interval_beats', 'interval_cv',
        'stair_density', 'pattern_switch_rate', 'direction_irregularity', 'movement_per_second', 'position_abs_mean']
print(f'{"特征":<26}{"定位":>8}{"硬抗":>8}{"读谱":>8}{"综合":>8} | 定位区分度')
for k in KEYS:
    vals = {}
    for cat in ['定位', '硬抗', '读谱', '综合']:
        vs = [f.get(k, 0) for f in [ff for c, ff in rows if c == cat]]
        vals[cat] = np.median(vs) if vs else 0
    # 区分度: 定位 vs 其他
    others = np.array([vals[c] for c in ['硬抗', '读谱', '综合']])
    sep = abs(vals['定位'] - others.mean())
    flag = '★' if sep > 0.5 * (others.std() + 1e-6) and sep > 0.3 else ''
    print(f'{k:<26}{vals["定位"]:>8.2f}{vals["硬抗"]:>8.2f}{vals["读谱"]:>8.2f}{vals["综合"]:>8.2f} {flag}')
print('DONE')
