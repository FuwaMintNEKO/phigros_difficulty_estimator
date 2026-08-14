# -*- coding: utf-8 -*-
"""抓 utils/index.ts 的 separateSpeedEvent"""
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
src = fetch(base + 'convert/utils/index.ts')
# 找 separateSpeedEvent
idx = src.find('separateSpeedEvent')
print(src[max(0,idx-200):idx+1500] if idx >= 0 else '未找到, 全文前2000:' + src[:2000])
print('DONE')