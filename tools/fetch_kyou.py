# -*- coding: utf-8 -*-
"""拉取 kyou.net.cn 数据: 官谱定数表 + 谱面特征投票

- GET /api/songs/songlist           官谱定数表 (id/标题/难度/BPM/曲包)
- GET /api/tags/tree                特征标签树 (读谱/操作/耐力 等分类)
- GET /api/tags/top-batch?          Top标签投票 (songIds批量, 无需登录)

输出到 data/kyou/ 下
"""
import os, sys, json, time, io
import urllib.request, urllib.parse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

BASE = 'https://kyou.net.cn'
OUT = os.path.join(_ROOT, 'data', 'kyou')
os.makedirs(OUT, exist_ok=True)
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


def get(url, retries=5):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def main():
    # 1. 定数表
    sl = get(BASE + '/api/songs/songlist')
    with open(os.path.join(OUT, 'songlist.json'), 'w', encoding='utf-8') as f:
        json.dump(sl, f, ensure_ascii=False, indent=1)
    songs = sl['data']
    print(f'[1] songlist: {len(songs)} songs')

    # 2. 标签树
    tg = get(BASE + '/api/tags/tree')
    with open(os.path.join(OUT, 'tags_tree.json'), 'w', encoding='utf-8') as f:
        json.dump(tg, f, ensure_ascii=False, indent=1)
    print(f'[2] tags_tree: {len(tg)} top-level')

    # 3. 全谱投票 (按难度批量)
    diffs = ['ez', 'hd', 'in', 'at', 'sp']
    ids = [s['id'] for s in songs]
    votes = {}
    for d in diffs:
        for i in range(0, len(ids), 150):
            chunk = ids[i:i + 150]
            q = urllib.parse.urlencode({'songIds': ','.join(chunk), 'difficulty': d, 'limitPerSong': 4})
            try:
                data = get(BASE + '/api/tags/top-batch?' + q)
                for k, v in data.items():
                    votes[f'{k}::{d}'] = v
                print(f'  [{d}] chunk {i//150+1}: +{len(data)}')
            except Exception as e:
                print(f'  [{d}] chunk {i//150+1} ERR {str(e)[:80]}')
            time.sleep(0.3)
    with open(os.path.join(OUT, 'votes.json'), 'w', encoding='utf-8') as f:
        json.dump(votes, f, ensure_ascii=False, indent=1)
    print(f'[3] votes: {len(votes)} song::diff entries')


if __name__ == '__main__':
    main()
