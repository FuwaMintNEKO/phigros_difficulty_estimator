# -*- coding: utf-8 -*-
"""全面统计 スタートリップ PE 文件命令结构"""
import re, os
from collections import Counter

p = r'C:\Users\NaNK\Downloads\スタートリップ(12.2).json'
with open(p, encoding='utf-8') as f:
    text = f.read()

cmds = Counter()
n_by_line = Counter()
bp_list = []
cp_by_line = Counter()
max_parts = 0
for raw in text.split('\n'):
    raw = raw.strip()
    if not raw or raw.startswith('#') or raw.startswith('&'):
        continue
    parts = raw.split()
    if not parts:
        continue
    cmd = parts[0]
    cmds[cmd] += 1
    if cmd in ('n1', 'n2', 'n3', 'n4'):
        if len(parts) >= 2:
            n_by_line[int(parts[1])] += 1
    elif cmd == 'bp':
        bp_list.append(parts)
    elif cmd == 'cp':
        if len(parts) >= 2:
            cp_by_line[int(parts[1])] += 1
    max_parts = max(max_parts, len(parts))

print('命令类型统计:', dict(cmds))
print('cp 参数最多列数:', max_parts)
print()
print('bp 命令 (变速段):')
for b in bp_list:
    print('  ', b)
print()
print('各线音符数 (n1-n4):')
for k in sorted(n_by_line):
    print(f'  线{k}: {n_by_line[k]}')
print(f'  合计: {sum(n_by_line.values())}')
print()
print('各线 cp 数 (线移动):')
for k in sorted(cp_by_line):
    print(f'  线{k}: {cp_by_line[k]}')
print(f'  合计: {sum(cp_by_line.values())}')
