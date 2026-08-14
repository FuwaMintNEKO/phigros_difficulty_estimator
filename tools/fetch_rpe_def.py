# -*- coding: utf-8 -*-
"""抓取权威源码: RPE note 类型定义"""
import os, sys, io, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return f'ERR: {e}'
base = 'https://raw.githubusercontent.com/187J3X1-114514/PhiChartRender/master/packages/phigros/src/core/chart/'
for f in ['types/rpe/index.ts', 'types/phigros/note.ts', 'convert/rephiedit.ts', 'convert/official.ts']:
    print(f'\n===== {f} =====')
    print(fetch(base + f)[:3000])
print('DONE')