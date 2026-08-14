# -*- coding: utf-8 -*-
"""B站评论区获取"""
import os, sys, io, json, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return f'ERR: {e}'
# 评论: oid=aid
r = fetch('https://api.bilibili.com/x/v2/reply?type=1&oid=116708824057951&sort=1&ps=30')
try:
    d = json.loads(r)
    if d.get('code') == 0:
        replies = d['data']['replies'] or []
        print(f'评论数: {len(replies)}')
        for rep in replies:
            user = rep['member']['uname']
            msg = rep['content']['message'].replace('\n', ' ')
            like = rep.get('like', 0)
            print(f'[{like}赞] {user}: {msg[:120]}')
            # 子回复
            sub = rep.get('replies') or []
            for s in sub[:3]:
                print(f'    └ {s["member"]["uname"]}: {s["content"]["message"][:80]}')
    else:
        print('API:', d.get('code'), d.get('message'))
except Exception as e:
    print('解析失败:', r[:300])
print('DONE')