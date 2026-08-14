# -*- coding: utf-8 -*-
"""全量拉取 phira 未上架(type=2) 谱面 → 筛4星+(rating>=0.8) → 补下载缺失到 json_unranked_4star/
拉取全部分页(不按difficulty截断), 保存元数据, 已有文件跳过
"""
import os, sys, json, io, time, zipfile, csv, urllib.request, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 'https://phira.5wyxi.com'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
OUT_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star')
os.makedirs(OUT_DIR, exist_ok=True)

def get(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if i == retries - 1: raise
            time.sleep(1.5 * (i + 1))

# ===== 1. 全量拉取 type=2 =====
allc = []
seen = set()
page = 1
MAX_PAGE = 600
print('=== 拉取 type=2 未上架全量 ===')
while page <= MAX_PAGE:
    try:
        data = get(f'{BASE}/chart?type=2&division=regular&order=-difficulty&pageNum=30&page={page}')
    except Exception as e:
        print(f'page{page} err {e}, 停止')
        break
    results = data.get('results') or data.get('result') or []
    if not results:
        print(f'page{page} 空, 结束')
        break
    added = 0
    for c in results:
        if c['id'] not in seen:
            seen.add(c['id']); allc.append(c); added += 1
    if page % 20 == 0:
        print(f'  page{page}: 累计 {len(allc)} 张')
    if added == 0:
        break
    page += 1
    time.sleep(0.15)
print(f'拉取结束: 共 {len(allc)} 张 (page={page})')

# 保存全量元数据
meta_path = os.path.join(_ROOT, 'data', 'phira', 'unranked_all.json')
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(allc, f, ensure_ascii=False)
print(f'元数据已存: {meta_path}')

# ===== 2. 筛 4星+ =====
sel = [c for c in allc if (c.get('rating') or 0) >= 0.8]
print(f'4星+(rating>=0.8): {len(sel)} / {len(allc)}')

# ===== 3. 已有 id =====
have = set(int(fn[:-5]) for fn in os.listdir(OUT_DIR) if fn.endswith('.json'))
need = [c for c in sel if c['id'] not in have]
print(f'已有: {len(have)}, 需下载: {len(need)}')

def download(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as e:
            if i == retries - 1: raise
            time.sleep(2)

def extract_json(raw, cid):
    z = zipfile.ZipFile(io.BytesIO(raw))
    cands = []
    for n in z.namelist():
        low = n.lower()
        if low.endswith('.json') and '/chart' not in low:
            cands.append((0 if low == f'{cid}.json' else 1, low, z.getinfo(n).file_size))
        elif low.endswith('.pec'):
            cands.append((2, low, z.getinfo(n).file_size))
    if not cands:
        raise ValueError('no chart file in zip')
    cands.sort(key=lambda t: (t[0], t[1]))
    return z.read(cands[0][1])

# ===== 4. 下载缺失 =====
ok = fail = 0
for i, c in enumerate(need):
    cid = c['id']
    dst = os.path.join(OUT_DIR, f'{cid}.json')
    try:
        raw = download(c['file'])
        data = extract_json(raw, cid)
        with open(dst, 'wb') as f:
            f.write(data)
        ok += 1
    except Exception as e:
        fail += 1
        print(f'  失败 {cid} {c.get("name","")[:20]}: {e}')
    if (i + 1) % 50 == 0:
        print(f'  下载进度: {i+1}/{len(need)} (成功{ok} 失败{fail})')
    time.sleep(0.25)
print(f'\n下载完成: 成功{ok} 失败{fail}, 目录共 {len(os.listdir(OUT_DIR))} 张')

# ===== 5. 更新清单 =====
def read_pred(path):
    preds = {}
    if not os.path.exists(path): return preds
    with open(path, encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f)
        head = next(rd)
        idx = {h: i for i, h in enumerate(head)}
        for c in rd:
            if len(c) < len(head): continue
            try: preds[int(c[idx['id']])] = {h: c[idx[h]] for h in head}
            except Exception: pass
    return preds
pred_map = read_pred(os.path.join(_ROOT, 'data', 'phira', 'v112_unranked_predictions_v2.csv'))

out_csv = os.path.join(_ROOT, 'data', 'phira', 'unranked_4star_list.csv')
with open(out_csv, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'name', 'level', 'difficulty', 'rating', 'ratingCount', 'pred', 'err', 'notes'])
    for c in sorted(sel, key=lambda x: -x['rating']):
        p = pred_map.get(c['id'], {})
        w.writerow([c['id'], c['name'], c['level'], c.get('difficulty'), round(c.get('rating', 0), 4), c.get('ratingCount', 0),
                    p.get('pred', ''), p.get('err', ''), p.get('notes', '')])
print(f'清单已更新: {out_csv} ({len(sel)} 行)')
print('DONE')
