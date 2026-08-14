# -*- coding: utf-8 -*-
"""抓 utils/utils.ts 的 separateSpeedEvent"""
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
src = fetch(base + 'convert/utils/utils.ts')
idx = src.find('separateSpeedEvent')
if idx >= 0:
    print(src[max(0,idx-100):idx+2500])
else:
    print('utils.ts 未找到, 找 speed 相关:')
    for m in ['speed', 'Speed']:
        for j in range(len(src)):
            if m in src[j:j+1]:
                pass
    # 打印所有含speed的行
    for i, ln in enumerate(src.split('\n')):
        if 'speed' in ln.lower():
            print(f'{i}: {ln.strip()[:130]}')
print('DONE')