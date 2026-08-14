# -*- coding: utf-8 -*-
"""并发下载未上架高难谱 (unranked_final_download.json), 解压 json 到 data/phira/json_unranked/
多线程并发, 断点续传
"""
import os, sys, json, time, io, zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator'
OUT = os.path.join(ROOT, 'data', 'phira', 'json_unranked')
os.makedirs(OUT, exist_ok=True)
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

sel = json.load(open(os.path.join(ROOT, 'data', 'phira', 'unranked_final_download.json'), encoding='utf-8'))
print(f'待下载 {len(sel)} 张', flush=True)

# 过滤已存在
todo = []
skip = 0
for c in sel:
    out_path = os.path.join(OUT, f"{c['id']}.json")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
        skip += 1
    else:
        todo.append(c)
print(f'已存在 {skip} 张, 需下载 {len(todo)} 张', flush=True)

def download_one(c):
    cid = c['id']
    out_path = os.path.join(OUT, f'{cid}.json')
    for attempt in range(3):
        try:
            req = urllib.request.Request(c['file'], headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
            if len(raw) < 100:
                raise ValueError('too small')
            z = zipfile.ZipFile(io.BytesIO(raw))
            cands = []
            for n in z.namelist():
                low = n.lower()
                if low.endswith('.json'):
                    cands.append((0, n != f'{cid}.json', n))
                elif low.endswith('.pec') and 'info' not in low:
                    cands.append((1, 0, n))
            if not cands:
                raise ValueError('no chart file')
            cands.sort(key=lambda t: (t[0], t[1], -z.getinfo(t[2]).file_size))
            with open(out_path, 'wb') as f:
                f.write(z.read(cands[0][2]))
            return ('ok', cid)
        except Exception as e:
            if attempt == 2:
                return ('fail', cid, str(e)[:60])
            time.sleep(1)

t0 = time.time()
ok = fail = 0
fails = []
with ThreadPoolExecutor(max_workers=12) as ex:
    futs = {ex.submit(download_one, c): c for c in todo}
    for i, fut in enumerate(as_completed(futs), 1):
        r = fut.result()
        if r[0] == 'ok':
            ok += 1
        else:
            fail += 1
            fails.append(r[1:])
        if i % 50 == 0:
            el = time.time() - t0
            print(f'  [{i}/{len(todo)}] ok={ok} fail={fail} 用时{el:.0f}s ({i/el:.1f}张/s)', flush=True)

print(f'\n完成: 成功 {ok}, 失败 {fail}, 跳过 {skip}')
for cid, err in fails[:20]:
    print(f'  FAIL {cid}: {err}')
