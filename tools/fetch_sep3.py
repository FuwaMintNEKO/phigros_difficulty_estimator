# -*- coding: utf-8 -*-
"""抓 rephiedit.ts 全文 (找 speedEvents 的 value 含义)"""
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
# 打印 100-180 行 (speed处理)
for i in range(100, 180):
    if i < len(lines):
        print(f'{i}: {lines[i]}')
print('DONE')