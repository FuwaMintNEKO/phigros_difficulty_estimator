# -*- coding: utf-8 -*-
"""查 phigros 相关开源解析器: PhigrosLibrary / PhiChartRender / phira 后端"""
import os, sys, io, urllib.request, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req, timeout=20)
        return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return f'ERR: {e}'
# 搜索 GitHub 仓库 (不带认证的search API 有限, 直接试已知仓库)
cands = [
    'https://api.github.com/repos/5wyxi/Phira',
    'https://api.github.com/repos/187J3X1-114514/PhiChartRender',
    'https://api.github.com/repos/7a9f1b2c/PhigrosLibrary',
]
for url in cands:
    r = fetch(url)
    try:
        d = json.loads(r)
        print(f'{d.get("full_name")}: {d.get("description")}')
        print(f'  language={d.get("language")} default_branch={d.get("default_branch")}')
    except:
        print(f'{url}: {r[:100]}')
print('DONE')