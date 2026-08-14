# -*- coding: utf-8 -*-
"""多面下落(ml)/判定线旋转/读谱 特征权重与分布"""
import os, sys, io, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])

print('=== MANUAL_FLAT 中相关特征权重 ===')
for fname, bl, co in app_mod.MANUAL_FLAT:
    if fname in ('multi_line_sim_events','jline_rotate_density','jline_movement_density','jline_disappear_density',
                 'speed_volatility','speed_event_density','hold_interference_index','hold_lock_weighted_per_hold',
                 'flash_hold_ratio','drag_per_sec','chord_jack_3plus_pairs'):
        p95 = app_mod.P95.get(fname, 0)
        print(f'  {fname:<34} bl={bl:<8} co={co:<8} P95={p95:.2f}')
print()
# 官谱/上架谱: 多面下落特征分布 (多线谱)
ml = np.array([r['feats'].get('multi_line_sim_events', 0) for r in ranked])
print(f'multi_line_sim_events: P50={np.median(ml):.1f} P90={np.percentile(ml,90):.1f} max={ml.max():.0f}')
print('ml>=50 的谱:')
for i in np.where(ml >= 50)[0]:
    print(f'  {ranked[i]["name"][:24]:<26} ml={ml[i]:.0f} diff={ds[i]:.1f}')
print()
# 官谱(无ml重)与多面谱的boost差异
for r in ranked:
    if r['id'] in (47264, 7516):
        f = dict(r['feats'])
        b, dims, _ = app_mod.compute_boost(f, 1.0, is_custom=True)
        print(f'{r["name"][:18]} ml={f.get("multi_line_sim_events",0):.0f} jrot={f.get("jline_rotate_density",0):.1f} jmov={f.get("jline_movement_density",0):.1f} 读谱贡献={dims["categories"].get("读谱",0):.3f} boost={b:.3f}')
print('DONE')