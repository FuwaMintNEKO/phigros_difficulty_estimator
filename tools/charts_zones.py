# -*- coding: utf-8 -*-
"""charts.json 分区结构: 上架 vs 特殊"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'charts.json')
with open(p, encoding='utf-8') as f:
    data = json.load(f)
print('分区:', list(data.keys()))
for zone, items in data.items():
    print(f'\n=== {zone} ===')
    print('type:', type(items), 'len:', len(items))
    if isinstance(items, list):
        print('首条:', items[0])
        # 找魔理沙
        for it in items:
            if isinstance(it, dict) and '魔理沙' in str(it.get('name', it.get('song', ''))):
                print('魔理沙:', it)
                break
    elif isinstance(items, dict):
        print('keys:', list(items.keys())[:5])
print('DONE')