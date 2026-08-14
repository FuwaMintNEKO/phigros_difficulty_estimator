# -*- coding: utf-8 -*-
"""PEC格式确认: # & 行的含义 + 官谱标准音符speed字段"""
import os, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'json', '7516.json')
with open(p, encoding='utf-8', errors='replace') as f:
    lines = f.read().splitlines()
print('总行数:', len(lines))
# 统计命令类型
from collections import Counter
cmds = Counter()
for ln in lines:
    ln = ln.strip()
    if not ln: continue
    cmds[ln.split()[0]] += 1
print('命令统计:', dict(cmds))
# 看 # 和 & 行的值分布
hash_vals = set(); amp_vals = set()
for ln in lines:
    ln = ln.strip()
    if ln.startswith('#'):
        hash_vals.add(ln)
    elif ln.startswith('&'):
        amp_vals.add(ln)
print('\n# 行样例(前5):', sorted(hash_vals)[:5], '共', len(hash_vals), '种')
print('& 行样例(前5):', sorted(amp_vals)[:5], '共', len(amp_vals), '种')
# 找一个长条 n2 的上下文 (前后5行)
for i, ln in enumerate(lines):
    if ln.strip().startswith('n2'):
        ctx = lines[max(0,i-4):i+2]
        print('\nn2 上下文:')
        for c in ctx: print('  ', c.strip())
        break
# 标准官谱音符 speed 字段?
import json, glob
std_files = glob.glob(os.path.join(_ROOT, 'data', 'phira', 'json', '*.json'))[:0]
# 用feats_cache里的官谱? 检查标准格式样例: 找formatVersion 3
cnt_sp = 0
for f in glob.glob(os.path.join(_ROOT, 'data', 'phira', 'json', '*.json'))[:200]:
    try:
        with open(f, encoding='utf-8', errors='replace') as fh:
            d = json.load(fh)
        if isinstance(d, dict) and 'judgeLineList' in d and d.get('formatVersion') == 3:
            for jl in d['judgeLineList']:
                for n in jl.get('notesAbove', []):
                    if 'speed' in n: cnt_sp += 1
    except Exception:
        pass
print('\n标准格式谱中带speed字段的音符数(前200文件):', cnt_sp)
print('DONE')