# -*- coding: utf-8 -*-
"""PhiChartRender 判定逻辑: drag vs tap 判定差异"""
import os, sys, io, urllib.request, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return f'ERR: {e}'
base = 'https://raw.githubusercontent.com/187J3X1-114514/PhiChartRender/master/packages/phigros/src/core/'
r = fetch(base + 'judgement/index.ts')
print('判定源码 (前4000字符):')
print(r[:4000])
print('DONE')