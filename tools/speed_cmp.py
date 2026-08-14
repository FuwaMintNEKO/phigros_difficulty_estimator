# -*- coding: utf-8 -*-
"""Bathin vs 官谱 speed特征详细对比"""
import os, sys, io, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])
KEYS = ['speed_event_count','speed_event_density','speed_mean','speed_std','speed_max','speed_min','speed_range','speed_volatility','speed_event_log_density','speed_volatility_log']
# 各谱
for r in ranked:
    if r['id'] in (7516, 47264, 15875, 59064):
        print(f'--- {r["name"][:20]} id={r["id"]} diff={round(r["diff"],1)} ---')
        for k in KEYS:
            print(f'  {k}={r["feats"].get(k,0):.3f}', end='')
        print()
# 官谱 P95
print('\nP95 (官谱训练):')
for k in KEYS:
    if k in app_mod.P95:
        print(f'  {k}: P95={app_mod.P95[k]:.3f}')
# ranked中speed_volatility最高15首
print('\nspeed_volatility 最高15首:')
idx = np.argsort(-np.array([r['feats'].get('speed_volatility',0) for r in ranked]))[:15]
for i in idx:
    print(f'  {ranked[i]["name"][:24]:<26} vol={ranked[i]["feats"].get("speed_volatility",0):.1f} diff={ds[i]:.1f} speedmax={ranked[i]["feats"].get("speed_max",0):.1f} speedrange={ranked[i]["feats"].get("speed_range",0):.1f}')
print('DONE')