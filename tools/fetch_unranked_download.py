# -*- coding: utf-8 -*-
"""下载未上架高难谱 (unranked_final_download.json), 解压 json 到 data/phira/json_unranked/
断点续传: 已存在则跳过
"""
import os, sys, json, time, io, zipfile
import urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator'
OUT = os.path.join(ROOT, 'data', 'phira', 'json_unranked')
os.makedirs(OUT, exist_ok=True)
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

sel = json.load(open(os.path.join(ROOT, 'data', 'phira', 'unranked_final_download.json'), encoding='utf-8'))
print(f'待下载 {len(sel)} 张', flush=True)

def download(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2)

ok = skip = 0
fail = []
t0 = time.time()
for i, c in enumerate(sel):
    cid = c['id']
    out_path = os.path.join(OUT, f'{cid}.json')
    if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
        skip += 1
        continue
    try:
        raw = download(c['file'])
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
        ok += 1
    except Exception as e:
        fail.append((cid, c.get('name', ''), str(e)[:60]))
    if (i + 1) % 20 == 0:
        el = time.time() - t0
        print(f'  [{i+1}/{len(sel)}] ok={ok} skip={skip} fail={len(fail)} 用时{el:.0f}s', flush=True)
    time.sleep(0.1)

print(f'\n完成: 成功 {ok}, 跳过 {skip}, 失败 {len(fail)}')
for cid, name, err in fail[:20]:
    print(f'  FAIL {cid} {name[:30]}: {err}')
