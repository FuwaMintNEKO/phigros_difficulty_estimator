# -*- coding: utf-8 -*-
"""探索特殊分区识别: 检查元数据字段"""
import os, sys, io, json, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# 1) unranked_all.json 元数据字段
meta = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'unranked_all.json'), encoding='utf-8'))
print('unranked_all 首条字段:', list(meta[0].keys()))
print('unranked_all 总数:', len(meta))
# 找魔理沙
for c in meta:
    if '魔理沙' in c.get('name', ''):
        print('\n魔理沙在unranked_all:', c)
# 2) 检查是否有 ranked 元数据文件
import glob
for f in glob.glob(os.path.join(_ROOT, 'data', 'phira', '*.json')):
    print('json文件:', os.path.basename(f))
print('DONE')