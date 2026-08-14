# -*- coding: utf-8 -*-
"""抓取 rephiedit.ts 中 note type 转换的核心代码"""
import os, sys, io, urllib.request, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return f'ERR: {e}'
base = 'https://raw.githubusercontent.com/187J3X1-114514/PhiChartRender/master/packages/phigros/src/core/chart/'
src = fetch(base + 'convert/rephiedit.ts')
lines = src.split('\n')
print('=== 前80行 (BPM处理) ===')
for i, ln in enumerate(lines[:80]):
    print(f'{i}: {ln}')
print('DONE')