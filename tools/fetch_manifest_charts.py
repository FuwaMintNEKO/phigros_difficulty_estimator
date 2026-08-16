# -*- coding: utf-8 -*-
"""下载 manifest17plus.tsv 中缺失的本地谱面 (正id从phira下载, 负id跳过)
输出: data/phira/json_unranked_4star/{id}.json
"""
import os, sys, json, io, time, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
BASE = 'https://phira.5wyxi.com'
H = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
OUT = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star')
os.makedirs(OUT, exist_ok=True)

def get_json(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=H)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2.0 * (i + 1))

def get_bytes(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=H)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2.0 * (i + 1))

rows = []
for l in open(os.path.join(_ROOT, 'data', 'phira', 'manifest17plus.tsv'), encoding='utf-8', errors='replace').read().splitlines()[2:]:
    p = l.split('	')
    if len(p) >= 3 and p[0].lstrip('-').isdigit():
        rows.append((int(p[0]), p[1], float(p[2])))

missing = []
for cid, name, diff in rows:
    if cid < 0:
        continue
    ok = os.path.exists(os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '%d.json' % cid))         or os.path.exists(os.path.join(_ROOT, 'data', 'phira', 'json_unranked', '%d.json' % cid))         or os.path.exists(os.path.join(_ROOT, 'data', 'phira', 'json', '%d.json' % cid))
    if not ok:
        missing.append((cid, name, diff))
print('待下载: %d 首' % len(missing), flush=True)

ok_n, fail = 0, []
for cid, name, diff in missing:
    try:
        d = get_json('%s/chart/%d' % (BASE, cid))
        furl = d.get('file')
        if not furl:
            fail.append((cid, 'no file url')); continue
        if not furl.startswith('http'):
            furl = BASE + furl
        data = get_bytes(furl)
        # 检查是否zip
        if data[:2] == b'PK':
            import zipfile, io as _io
            z = zipfile.ZipFile(_io.BytesIO(data))
            data = None
            for n in z.namelist():
                if n.endswith('.json'):
                    data = z.read(n)
                    break
            if data is None:
                for n in z.namelist():
                    if n.endswith('.pec'):   # PE文本打包 (如Ultimate Force/Staring at star)
                        data = z.read(n)
                        break
            if data is None:
                fail.append((cid, 'zip无谱面'))
                continue
        with open(os.path.join(OUT, '%d.json' % cid), 'wb') as f:
            f.write(data)
        ok_n += 1
        print('#%d %-30s %.1f -> %d bytes' % (cid, name[:30], diff, len(data)), flush=True)
    except Exception as e:
        fail.append((cid, str(e)[:80]))
        print('FAIL #%d %s: %s' % (cid, name[:20], str(e)[:60]), flush=True)
    time.sleep(0.3)
print('下载成功 %d, 失败 %d' % (ok_n, len(fail)), flush=True)
for cid, e in fail:
    print('  #%d: %s' % (cid, e), flush=True)
