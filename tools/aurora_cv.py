# -*- coding: utf-8 -*-
"""Aurora 原始 cv 行: 找1711109.4值对应行"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'json', '10203.json')
with open(p, encoding='utf-8', errors='replace') as f:
    lines = f.read().splitlines()
cnt = 0
for ln in lines:
    ln = ln.strip()
    if ln.startswith('cv'):
        parts = ln.split()
        if len(parts) >= 4 and abs(float(parts[3])) > 100:
            cnt += 1
            if cnt <= 8:
                print('大值cv行:', parts)
print('大值cv行总数:', cnt)
# 正常cv行样例
cnt2 = 0
for ln in lines:
    ln = ln.strip()
    if ln.startswith('cv'):
        parts = ln.split()
        if len(parts) >= 4 and abs(float(parts[3])) <= 100:
            cnt2 += 1
            if cnt2 <= 5:
                print('正常cv行:', parts)
print('正常cv行总数:', cnt2)
print('DONE')