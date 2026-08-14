# -*- coding: utf-8 -*-
"""16-16.5 vs 16.5+ 特征对比: 找区分高难段的特征"""
import os, sys, numpy as np, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])

# 候选区分特征 (排除整数定数谱)
cands = ['hold_count','total_notes','hold_ratio','hold_lock_weighted','per_hold','per_sec',
         'fast_ms_050_ratio','fast_ms_100_ratio','fast_ms_150_ratio','interaction_ms_run',
         'multi_finger_3plus_events','multi_line_sim_events','weighted_mf_score_per_sec',
         'above_avg_density_mean','real_core_notes_per_second','real_notes_per_second',
         'cross_hand_density','lane_switch_density','eff_peak_tps_1s','eff_avg_tps_1s',
         'global_jack_count','chord_alternation_rate','discrete_mf_ratio','duration_sec',
         'tracks_active_sec','tracks_4plus_sec','tracks_5plus_sec','tracks_6plus_sec',
         'bpm','bpm_max','bpm_change_count','rest_ratio','slider_ratio']
def getf(r, k):
    v = r['feats'].get(k, 0)
    return v
A = [r for i,r in enumerate(ranked) if 16 <= ds[i] < 16.5]
B = [r for i,r in enumerate(ranked) if ds[i] >= 16.5]
print(f'A=16-16.5 n={len(A)}  B=16.5+ n={len(B)}')
print(f'{"特征":<32}{"A均值":>10}{"B均值":>10}{"B/A":>8}')
for k in cands:
    va = np.array([getf(r,k) for r in A]); vb = np.array([getf(r,k) for r in B])
    if va.std() == 0 and vb.std() == 0:
        print(f'{k:<32}{va.mean():>10.3f}{vb.mean():>10.3f}{"":>8}')
        continue
    # 简单判别力: 两均值差 / 合并std
    m = (va.mean()-vb.mean())/np.sqrt(va.var()+vb.var())
    print(f'{k:<32}{va.mean():>10.3f}{vb.mean():>10.3f}{m:>8.2f}')
# hold加成分布: 两段各多少谱触发0.25/0.4/0.6
for k, lbl in [('hold_ratio','hold_ratio')]:
    va = np.array([getf(r,k) for r in A]); vb = np.array([getf(r,k) for r in B])
    for thr in [0.25,0.4,0.6]:
        print(f'{lbl}>= {thr}: A={np.mean(va>=thr):.2f} B={np.mean(vb>=thr):.2f}')
# 官谱 vs 上架谱构成
print('\n官谱(kyou共识)构成:')
for lbl, seg in [('16-16.5', A), ('16.5+', B)]:
    n_off = sum(1 for r in seg if r.get('is_official'))
    print(f'  {lbl}: 官谱 {n_off}/{len(seg)}')
print('DONE')