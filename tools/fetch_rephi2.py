# -*- coding: utf-8 -*-
"""抓 rephiedit.ts note 转换段"""
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
print(f'总行数: {len(lines)}')
# 找 isMulti / holdLength / noteType / type 处理
for i, ln in enumerate(lines):
    if any(k in ln for k in ['isMulti', 'holdLength', 'noteType', 'type =', 'type===', 'case 1', 'case 2', 'case 3', 'case 4', 'Tap', 'Drag', 'Flick', 'Hold']):
        print(f'{i}: {ln.strip()[:120]}')
print('DONE')