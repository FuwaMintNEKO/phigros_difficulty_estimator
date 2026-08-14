# -*- coding: utf-8 -*-
"""RPE type语义定论: 看已知官谱高仿(夢降日/DerSchneid)的 type分布"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from collections import Counter

# 官谱高仿 (配置与官谱一致, type语义确定)
cases = [
    ('夢降日高仿', os.path.join(_ROOT, 'tools', '_tmp_dl_charts', '夢の降る日に', '5333883479687925.json')),
    ('DerSchneid高仿', os.path.join(_ROOT, 'tools', '_tmp_dl_charts', 'Der Schneid(1)', '1903581575578621.json')),
    ('Melodiniq', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')),
]
for nm, p in cases:
    raw = json.load(open(p, encoding='utf-8'))
    meta = raw.get('META', {})
    print(f'\n{nm} (RPEVersion={meta.get("RPEVersion")}):')
    cnt = Counter()
    for jl in raw.get('judgeLineList', []):
        for n in jl.get('notes', []):
            cnt[n.get('type')] += 1
    print('  type分布:', dict(cnt))
# 官谱原谱标准类型分布
for nm, p in [('夢降日官谱IN', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')),
              ('DerSchneid官谱AT', os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json'))]:
    raw = json.load(open(p, encoding='utf-8'))
    cnt = Counter()
    for jl in raw.get('judgeLineList', []):
        for n in jl.get('notesAbove', []) + jl.get('notesBelow', []):
            cnt[n.get('type')] += 1
    print(f'\n{nm} 标准type分布 (1=tap 2=drag 3=hold 4=flick): {dict(cnt)}')
print('DONE')