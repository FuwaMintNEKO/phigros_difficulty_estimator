# -*- coding: utf-8 -*-
"""RPE type语义: 多谱对比 type2/type4 的 endTime/特征"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# 已下载 unranked RPE 谱中抽样检查 type 与 endTime 的关系
import glob
files = glob.glob(os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '*.json'))[:400]
from collections import Counter
# type → 有endTime比例
t_end = Counter(); t_cnt = Counter()
for f in files:
    try:
        raw = json.load(open(f, encoding='utf-8'))
        if 'judgeLineList' not in raw: continue
        for jl in raw.get('judgeLineList', []):
            for n in jl.get('notes', []):
                ty = n.get('type')
                t_cnt[ty] += 1
                if n.get('endTime') is not None:
                    t_end[ty] += 1
    except Exception:
        pass
print('RPE type 分布 (抽样400谱):')
for ty in sorted(t_cnt):
    print(f'  type{ty}: {t_cnt[ty]} 个, 带endTime={t_end[ty]} ({t_end[ty]/t_cnt[ty]*100:.0f}%)')
print('\n结论: type带endTime比例高 = Hold语义')
print('DONE')