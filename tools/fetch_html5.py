# -*- coding: utf-8 -*-
"""Phigros drag 判定机制 (bilibili/知乎资料)"""
import os, sys, io, urllib.request, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return f'ERR: {e}'
# 查 phigros-html5 (社区权威实现) 的判定代码
r = fetch('https://raw.githubusercontent.com/yuameshi/phigros-html5/master/js/判定.js')
print('判定.js:', r[:500] if 'ERR' not in r else r)
# 或尝试其他路径
for path in ['js/judge.js', 'js/main.js', 'js/note.js']:
    r2 = fetch('https://raw.githubusercontent.com/yuameshi/phigros-html5/master/' + path)
    if 'ERR' not in r2:
        # 搜 drag
        for m in re.finditer(r'drag|Drag', r2):
            s = max(0, m.start()-100)
            print(f'\n[{path}] ...{r2[s:m.end()+200]}...')
            break
print('DONE')