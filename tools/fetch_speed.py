# -*- coding: utf-8 -*-
"""抓 PhiChartRender speedEvents 转换逻辑"""
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
src = fetch(base + 'convert/rephiedit.ts')
lines = src.split('\n')
# 找 speedEvents 处理段
for i, ln in enumerate(lines):
    if 'speedEvents' in ln or 'speedAnim' in ln or 'speed' in ln.lower() and ('event' in ln.lower() or 'value' in ln.lower()):
        print(f'{i}: {ln.strip()[:130]}')
print('DONE')