# -*- coding: utf-8 -*-
"""Phigros drag 判定机制 (来自官方/权威资料)"""
import os, sys, io, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return f'ERR: {e}'
# Phigros fandom: Game Mechanics (drag判定)
r = fetch('https://phigros.fandom.com/wiki/Game_Mechanics')
# 提取 drag 相关内容
import re
text = re.sub(r'<[^>]+>', ' ', r)
text = re.sub(r'\s+', ' ', text)
for kw in ['Drag', 'drag']:
    for m in re.finditer(kw, text):
        s = max(0, m.start()-150)
        seg = text[s:m.end()+250]
        if 'judge' in seg.lower() or 'touch' in seg.lower() or 'tap' in seg.lower():
            print(f'[{kw}] ...{seg}...')
            print('---')
            break
print('DONE')