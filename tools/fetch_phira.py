# -*- coding: utf-8 -*-
"""拉取 phira 自制谱元数据 (上架/特殊分区, 全部页)

- GET /chart?type=0&order=-rating&pageNum=30&page=N   上架谱 (ranked)
- GET /chart?type=1 ...                               特殊谱 (SP等)
输出 data/phira/charts.json  (含 id/name/level/difficulty/rating/file...)
"""
import os, sys, json, time, io
import urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(_ROOT, 'data', 'phira')
os.makedirs(OUT, exist_ok=True)
BASE = 'https://phira.5wyxi.com'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


def get(url, retries=5):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def fetch_type(t, name):
    out = []
    page = 1
    while True:
        url = f'{BASE}/chart?type={t}&order=-rating&pageNum=30&page={page}'
        data = get(url)
        results = data.get('results') or data.get('result') or []
        out.extend(results)
        total = data.get('count', 0)
        if len(out) >= total or not results:
            break
        page += 1
        time.sleep(0.3)
    print(f'[{name}] 拉取 {len(out)} 张')
    return out


def main():
    charts = {}
    for t, name in [(0, '上架'), (1, '特殊')]:
        charts[name] = fetch_type(t, name)
    with open(os.path.join(OUT, 'charts.json'), 'w', encoding='utf-8') as f:
        json.dump(charts, f, ensure_ascii=False, indent=1)
    print('已保存 data/phira/charts.json')


if __name__ == '__main__':
    main()
