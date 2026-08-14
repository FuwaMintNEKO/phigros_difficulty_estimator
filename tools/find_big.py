# -*- coding: utf-8 -*-
"""找 1711109.4 与 200000000 出现位置"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
for fname, pat in [('10203.json', '1711109'), ('37193.json', '200000000'), ('8565.json', '1711109')]:
    p = os.path.join(_ROOT, 'data', 'phira', 'json', fname)
    with open(p, encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    print(f'--- {fname} 搜索 {pat} ---')
    cnt = 0
    for i, ln in enumerate(lines):
        if pat in ln:
            cnt += 1
            if cnt <= 4:
                print(f'  行{i}: {ln.strip()[:150]}')
    print(f'  命中 {cnt} 行')
print('DONE')