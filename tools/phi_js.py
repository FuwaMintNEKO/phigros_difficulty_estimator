# -*- coding: utf-8 -*-
"""PhiChartRender: 全部js文件列表"""
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
try:
    d = json.loads(r)
    for t in d.get('tree', []):
        if t['path'].endswith('.js') and 'node_modules' not in t['path']:
            print(t['path'])
except Exception as e:
    print('ERR', e, r[:200])
print('DONE')