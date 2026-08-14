# -*- coding: utf-8 -*-
"""抓取 phigros-html5 源码: note type 定义"""
import os, sys, io, urllib.request, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req, timeout=20)
        return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return f'ERR: {e}'
# GitHub API 找仓库树
r = fetch('https://api.github.com/repos/yuameshi/phigros-charts-repo/git/trees/master?recursive=1')
try:
    d = json.loads(r)
    print('tree items:', len(d.get('tree', [])))
    for t in d.get('tree', []):
        if any(k in t['path'].lower() for k in ['parse', 'chart', 'note', 'loader']):
            print(' ', t['path'])
except Exception as e:
    print('ERR', e, r[:200])
print('DONE')