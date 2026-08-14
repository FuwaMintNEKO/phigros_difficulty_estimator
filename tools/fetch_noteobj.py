# -*- coding: utf-8 -*-
"""RPE type4 的视觉: 查 Phigros 资源/渲染 (drag 是黄色还是蓝色)"""
import os, sys, io, urllib.request, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return f'ERR: {e}'
# PhiChartRender note.ts: 看note渲染
base = 'https://raw.githubusercontent.com/187J3X1-114514/PhiChartRender/master/packages/phigros/src/core/chart/'
r = fetch(base + 'object/note.ts')
print('note.ts 前2000字符:')
print(r[:2000])
print('DONE')