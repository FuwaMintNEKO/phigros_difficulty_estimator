# -*- coding: utf-8 -*-
"""PhiChartRender: 找 note type 定义"""
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
        p = t['path'].lower()
        if any(k in p for k in ['chart.js', 'note.js', 'parser', 'loader', 'util']) and t['path'].endswith('.js'):
            print(t['path'])
except Exception as e:
    print('ERR', e, r[:200])
print('DONE')