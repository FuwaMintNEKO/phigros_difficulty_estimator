# -*- coding: utf-8 -*-
"""Bathin 原始文件: 前几行看格式"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'json', '7516.json')
with open(p, 'rb') as f:
    head = f.read(500)
print(repr(head[:400]))
print()
# 看文件大小
print('大小:', os.path.getsize(p))
print('DONE')