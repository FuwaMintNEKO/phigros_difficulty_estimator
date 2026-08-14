# -*- coding: utf-8 -*-
"""B站API: 视频信息 + 热门评论"""
import os, sys, io, json, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
def fetch(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        return f'ERR: {e}'
# 视频信息
r = fetch('https://api.bilibili.com/x/web-interface/view?bvid=BV1d6Et6WE7C')
try:
    d = json.loads(r)
    if d.get('code') == 0:
        data = d['data']
        print('标题:', data.get('title'))
        print('UP主:', data.get('owner', {}).get('name'))
        print('简介:', (data.get('desc') or '')[:200])
        print('aid:', data.get('aid'), 'cid:', data.get('cid'))
        print('播放:', data.get('stat', {}).get('view'))
    else:
        print('API返回:', d.get('code'), d.get('message'))
except Exception as e:
    print('解析失败:', r[:300])
print('DONE')