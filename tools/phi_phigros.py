# -*- coding: utf-8 -*-
"""PhiChartRender: phigros 包源码"""
import os, sys, io, urllib.request, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req, timeout=20)
        return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return f'ERR: {e}'
r = fetch('https://api.github.com/repos/187J3X1-114514/PhiChartRender/git/trees/master?recursive=1')
d = json.loads(r)
for t in d.get('tree', []):
    p = t['path']
    if p.startswith('packages/phigros') and p.endswith(('.js', '.ts', '.vue')):
        print(p)
print('DONE')