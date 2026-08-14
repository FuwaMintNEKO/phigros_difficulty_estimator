# -*- coding: utf-8 -*-
"""查 speed_max 巨大值谱的原始格式 (颜/Remember Our Summer/Aurora)"""
import os, sys, io, pickle, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
# 找这些谱的id
for r in ranked:
    if r['name'].startswith('颜') or r['name'].startswith('Remember Our') or r['name'].startswith('Aurora'):
        print(r['id'], r['name'], 'diff=', round(r['diff'],1), 'speed_max=', r['feats'].get('speed_max',0), 'speed_volatility=', r['feats'].get('speed_volatility',0))
        # 找原始文件
        p = os.path.join(_ROOT, 'data', 'phira', 'json', f"{r['id']}.json")
        if os.path.exists(p):
            with open(p, 'rb') as f:
                head = f.read(200)
            print('  文件头:', repr(head[:150]))
        print()
print('DONE')