# -*- coding: utf-8 -*-
"""官谱(kyou 310首) jline分布 vs 模型P95"""
import os, sys, io, json, numpy as np, pickle, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
with open(os.path.join(_ROOT, 'data', 'phira', 'kyou_tags.json'), encoding='utf-8') as f:
    kt = json.load(f)
kyou_names = [norm for item in kt for norm in [re.sub(r'[^0-9a-z一-鿿]', '', (item['song'] or '').lower())]]
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])

def norm(s):
    return re.sub(r'[^0-9a-z一-鿿]', '', (s or '').lower())
off = []
for i, r in enumerate(ranked):
    nm = norm(r['name'])
    if any(nm == kk or kk in nm or nm in kk for kk in kyou_names):
        off.append(i)
off = np.array(off)
print(f'官谱匹配: {len(off)}/{len(ranked)}')
for k in ['jline_movement_density', 'jline_rotate_density', 'jline_disappear_density', 'multi_line_sim_events']:
    vals_off = np.array([ranked[i]['feats'].get(k, 0) for i in off])
    vals_all = np.array([r['feats'].get(k, 0) for r in ranked])
    print(f'{k:<28} 官谱P95={np.percentile(vals_off,95):8.1f} 全部P95={np.percentile(vals_all,95):8.1f} 模型P95={app_mod.P95.get(k,0):8.1f}')
print('DONE')