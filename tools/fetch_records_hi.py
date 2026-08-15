# -*- coding: utf-8 -*-
"""A计划验证: 抓取未上架高难谱(预测17~19.5)的游玩记录
- 选谱: v1210_unranked_4star_predictions.csv 中 pred∈[17,19.5] 且 records>=300 的谱
- 分层取样: 17.x取15首 / 18.x取15首 / 19+取10首; 另补锚点对照谱
- 每首抓至多 300 条记录 (pageNum=30, 最多10页), 存 data/phira/records_hi/{id}.json
- 限速 ~0.35s/请求, 失败重试
"""
import os, sys, json, io, csv, time, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
BASE = 'https://phira.5wyxi.com'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
OUT = os.path.join(_ROOT, 'data', 'phira', 'records_hi')
os.makedirs(OUT, exist_ok=True)

def get(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2.0 * (i + 1))

# 1. 读预测CSV, 选候选
rows = list(csv.reader(open(os.path.join(_ROOT, 'data', 'phira', 'v1210_unranked_4star_predictions.csv'), encoding='utf-8-sig')))
hdr = rows[0]
cands = []
for r in rows[1:]:
    try:
        cid, name, pred = int(r[0]), r[1], float(r[4])
    except Exception:
        continue
    if 17.0 <= pred < 19.5:
        cands.append((cid, name, pred))
print('候选(17<=pred<19.5): %d 首' % len(cands), flush=True)

# 2. 查 count, 过滤 records>=300
withcount = []
for cid, name, pred in cands:
    try:
        d = get('%s/record/query/%d?page=1&pageNum=4' % (BASE, cid))
        cnt = d.get('count', 0) if isinstance(d, dict) else len(d)
    except Exception:
        cnt = 0
    if cnt >= 300:
        withcount.append((cid, name, pred, cnt))
    if (len(withcount) + 0) % 20 == 0:
        print('  count查询进度 %d/%d' % (len(withcount), len(cands)), flush=True)
    time.sleep(0.3)
print('records>=300: %d 首' % len(withcount), flush=True)

# 3. 分层取样
def pick(lo, hi, n):
    sel = sorted([c for c in withcount if lo <= c[2] < hi], key=lambda c: -c[3])[:n]
    return sel
chosen = []
chosen += pick(17.0, 18.0, 15)
chosen += pick(18.0, 19.0, 15)
chosen += pick(19.0, 19.5, 10)
# 锚点对照补入
ANCHORS = [41242, 44705, 42113, 70220, 52543, 294, 60137]
have = {c[0] for c in chosen}
for aid in ANCHORS:
    if aid in have:
        continue
    try:
        d = get('%s/record/query/%d?page=1&pageNum=4' % (BASE, aid))
        cnt = d.get('count', 0) if isinstance(d, dict) else len(d)
        nm = 'anchor'
        chosen.append((aid, nm, 0.0, cnt))
        time.sleep(0.3)
    except Exception:
        pass
print('选中 %d 首:' % len(chosen), flush=True)
for c in chosen:
    print('  #%-7d pred=%-6s records=%-5d' % (c[0], c[2], c[3]), flush=True)

# 4. 抓记录
for cid, name, pred, cnt in chosen:
    recs = []
    page = 1
    while len(recs) < min(cnt, 300):
        d = get('%s/record/query/%d?page=%d&pageNum=30' % (BASE, cid, page))
        batch = d.get('result') or d.get('results') or (d if isinstance(d, list) else [])
        if not batch:
            break
        recs.extend(batch)
        page += 1
        if len(batch) < 30:
            break
        time.sleep(0.35)
    with open(os.path.join(OUT, '%d.json' % cid), 'w', encoding='utf-8') as f:
        json.dump({'chart': cid, 'count': cnt, 'records': recs[:300]}, f, ensure_ascii=False)
    print('#%d 抓到 %d 条' % (cid, len(recs)), flush=True)
    time.sleep(0.35)
print('ALL DONE', flush=True)
