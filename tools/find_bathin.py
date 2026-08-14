# -*- coding: utf-8 -*-
"""Bathin 原始谱面: 检查speed字段"""
import os, sys, io, json, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# 找 ranked json 目录
for d in ['json_ranked', 'json_rank', 'ranked']:
    p = os.path.join(_ROOT, 'data', 'phira', d)
    if os.path.isdir(p):
        print('目录:', p, len(os.listdir(p)))
# 直接找 7516.json
for root, dirs, files in os.walk(os.path.join(_ROOT, 'data', 'phira')):
    for f in files:
        if f in ('7516.json', '7516.rpe', '7516.zip'):
            print('找到:', os.path.join(root, f))
print('DONE')