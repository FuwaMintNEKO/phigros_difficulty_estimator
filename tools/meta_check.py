# -*- coding: utf-8 -*-
"""高仿谱的 level/difficulty 元数据"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
for nm, p in [('夢降日', os.path.join(DL, '夢の降る日に', '5333883479687925.json')),
              ('DerSchneid', os.path.join(DL, 'Der Schneid(1)', '1903581575578621.json'))]:
    with open(p, 'rb') as f:
        cd, raw = load_chart_from_bytes(f.read())
    meta = cd.get('META', {})
    print(f'\n{nm}:')
    for k in ['name', 'level', 'difficulty', 'charter', 'song', 'RPEVersion']:
        print(f'  {k}: {meta.get(k, cd.get(k, "?"))}')
    # info.yml 看看
    info = os.path.join(os.path.dirname(p), 'info.yml')
    if os.path.exists(info):
        with open(info, encoding='utf-8') as f:
            print('  info.yml:', f.read()[:300])
print('DONE')