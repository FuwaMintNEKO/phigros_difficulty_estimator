# -*- coding: utf-8 -*-
"""筛选 phira 未上架(type=2) + 常规(division=regular) 的高难谱(16-18) + 高评分(>=0.84)
拉取 order=-difficulty 直到 difficulty<16, 本地筛选, 保存元数据
"""
import json, io, sys, urllib.request, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
BASE = 'https://phira.5wyxi.com'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
OUT = r'data\phira\unranked_regular_16plus.json'

def get(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))

allc = []
seen = set()
page = 1
stopped = False
while page <= 80:
    try:
        data = get(f'{BASE}/chart?type=2&division=regular&order=-difficulty&pageNum=30&page={page}')
    except Exception as e:
        print(f'page{page} err {e}, 停止')
        break
    results = data.get('results') or data.get('result') or []
    if not results:
        break
    # 记录本页 difficulty 范围
    diffs = [c.get('difficulty', 0) for c in results]
    dmin, dmax = min(diffs), max(diffs)
    # 去重添加
    added = 0
    for c in results:
        if c['id'] not in seen:
            seen.add(c['id']); allc.append(c); added += 1
    if page % 5 == 0 or dmin <= 16:
        print(f'  page{page}: 本页 diff {dmin:.1f}~{dmax:.1f}, 累计 {len(allc)} 张')
    # 如果本页最低 difficulty < 16, 说明已经扫过 16-18 区间
    if dmin < 16:
        stopped = True
        break
    page += 1
    time.sleep(0.15)

print(f'\n拉取结束: 共 {len(allc)} 张 (停止于 page{page}, stopped={stopped})')

# 筛选: 16<=diff<=18 + rating>=0.84 + tags含regular
def is_regular(c):
    tags = c.get('tags') or []
    if isinstance(tags, str):
        tags = [tags]
    return 'regular' in tags

sel = [c for c in allc if 16 <= c.get('difficulty', 0) <= 18
       and c.get('rating', 0) >= 0.84
       and is_regular(c)]

print(f'\n筛选结果 (16<=diff<=18, rating>=0.84, regular): {len(sel)} 张')
print(f'  其中 rating>=0.9 (4.5星): {sum(1 for c in sel if c["rating"]>=0.9)}')
print(f'  其中 0.84<=rating<0.9 (4.2-4.5星): {sum(1 for c in sel if 0.84<=c["rating"]<0.9)}')

# 按 rating 降序保存
sel.sort(key=lambda c: -c.get('rating', 0))
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(sel, f, ensure_ascii=False, indent=1)

print(f'\n已保存 {OUT}')
print('\n前 25 张 (rating 降序):')
for c in sel[:25]:
    print(f'  {c["id"]} {c.get("name","")[:26]:<26} diff={c.get("difficulty"):.1f} rating={c.get("rating"):.4f} rc={c.get("ratingCount")} lv={c.get("level","")}')
