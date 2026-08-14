# -*- coding: utf-8 -*-
"""B站评论区 v3: 旧版API ps=20"""
import os, sys, io, json, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return f'ERR: {e}'
r = fetch('https://api.bilibili.com/x/v2/reply?type=1&oid=116708824057951&sort=1&ps=20&pn=1')
try:
    d = json.loads(r)
    if d.get('code') == 0:
        replies = (d.get('data') or {}).get('replies') or []
        print(f'评论数: {len(replies)}')
        for rep in replies:
            user = rep['member']['uname']
            msg = rep['content']['message'].replace('\n', ' ')
            like = rep.get('like', 0)
            print(f'[{like}赞] {user}: {msg[:130]}')
    else:
        print('API:', d.get('code'), d.get('message'))
except Exception as e:
    print('解析失败:', r[:300])
print('DONE')