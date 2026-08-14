# -*- coding: utf-8 -*-
"""B站评论区 (修正参数)"""
import os, sys, io, json, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return f'ERR: {e}'
# 新版评论API: reply/wise
r = fetch('https://api.bilibili.com/x/v2/reply/wise/main?oid=116708824057951&type=1&mode=3&ps=30')
try:
    d = json.loads(r)
    if d.get('code') == 0:
        data = d.get('data', {})
        replies = (data.get('replies') or []) if isinstance(data, dict) else []
        print(f'评论数: {len(replies)}')
        for rep in replies[:25]:
            user = rep['member']['uname']
            msg = rep['content']['message'].replace('\n', ' ')
            like = rep.get('like', 0)
            print(f'[{like}赞] {user}: {msg[:130]}')
    else:
        print('API:', d.get('code'), d.get('message'), r[:200])
except Exception as e:
    print('解析失败:', r[:300])
print('DONE')