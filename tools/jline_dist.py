# -*- coding: utf-8 -*-
"""jline特征分布: 官谱P95是否被瞬移谱污染"""
import os, sys, io, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])

for k in ['jline_movement_density', 'jline_rotate_density', 'jline_disappear_density', 'speed_volatility', 'multi_line_sim_events', 'hold_interference_index']:
    vals = np.array([r['feats'].get(k, 0) for r in ranked])
    p95 = app_mod.P95.get(k, 0)
    print(f'{k:<28} P50={np.median(vals):8.1f} P75={np.percentile(vals,75):8.1f} P90={np.percentile(vals,90):8.1f} P95(模型)={p95:10.1f} P95(ranked)={np.percentile(vals,95):10.1f}')
    # 模型P95的 percentile in ranked
    pct = np.mean(vals <= p95) * 100
    print(f'    模型P95在ranked中的百分位: {pct:.0f}%  max={vals.max():.0f}')
print()
# Feeling Blue 各特征值 vs 触发阈值
print('Feeling Blue 触发检查:')
for r in ranked:
    if r['id'] == 47264:
        for k in ['jline_movement_density', 'jline_rotate_density', 'jline_disappear_density', 'speed_volatility', 'multi_line_sim_events', 'hold_interference_index', 'drag_per_sec', 'note_clutter_ratio']:
            v = r['feats'].get(k, 0)
            p95 = app_mod.P95.get(k, 0)
            thr = max(p95 * 0.55, 0)
            print(f'  {k}: v={v:.1f} 阈值={thr:.1f} {"触发" if v > thr else "不触发"}')
        break
print('DONE')