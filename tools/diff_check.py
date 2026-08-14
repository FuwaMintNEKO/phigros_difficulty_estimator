# -*- coding: utf-8 -*-
"""查官谱真实定数: 夢の降る日に / DerSchneid"""
import os, sys, io, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
with open(p, encoding='utf-8') as f:
    rd = csv.reader(f, delimiter='	')
    head = next(rd)
    print('表头:', head)
    for row in rd:
        line = '	'.join(row)
        if '夢' in line or 'Schneid' in line or 'DerSchneid' in line:
            print(row)
print('DONE')