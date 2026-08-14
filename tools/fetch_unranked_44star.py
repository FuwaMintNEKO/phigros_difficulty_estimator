# -*- coding: utf-8 -*-
"""4.4星+(rating>=0.88) 未上架谱 并发下载 → json_unranked_4star/
元数据: data/phira/unranked_all.json (8634张已全量拉取)
已有文件跳过; 8线程并发; 更新清单
"""
import os, sys, json, io, time, zipfile, csv, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 'https://phira.5wyxi.com'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
OUT_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star')
os.makedirs(OUT_DIR, exist_ok=True)

meta = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'unranked_all.json'), encoding='utf-8'))
sel = [c for c in meta if (c.get('rating') or 0) >= 0.88]
have = set(int(fn[:-5]) for fn in os.listdir(OUT_DIR) if fn.endswith('.json'))
need = [c for c in sel if c['id'] not in have]
print(f'4.4星+: {len(sel)}, 已有: {len(have)}, 需下载: {len(need)}')

def download_one(c):
    cid = c['id']
    dst = os.path.join(OUT_DIR, f'{cid}.json')
    if os.path.exists(dst):
        return (cid, True, 'skip')
    for i in range(3):
        try:
            req = urllib.request.Request(c['file'], headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            z = zipfile.ZipFile(io.BytesIO(raw))
            cands = []
            for n in z.namelist():
                low = n.lower()
                if low.endswith('.json') and '/chart' not in low:
                    cands.append((0 if low == f'{cid}.json' else 1, low, z.getinfo(n).file_size))
                elif low.endswith('.pec'):
                    cands.append((2, low, z.getinfo(n).file_size))
            if not cands:
                return (cid, False, 'no chart file')
            cands.sort(key=lambda t: (t[0], t[1]))
            data = z.read(cands[0][1])
            with open(dst, 'wb') as f:
                f.write(data)
            return (cid, True, 'ok')
        except Exception as e:
            if i == 2:
                return (cid, False, str(e)[:80])
            time.sleep(1.5)
    return (cid, False, 'unknown')

N_WORKERS = 8
ok = fail = 0
t0 = time.time()
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futs = {ex.submit(download_one, c): c for c in need}
    done = 0
    for fut in as_completed(futs):
        cid, success, msg = fut.result()
        done += 1
        if success: ok += 1
        else:
            fail += 1
            if fail <= 10:
                print(f'  失败 {cid} {futs[fut].get("name","")[:24]}: {msg}')
        if done % 200 == 0:
            el = time.time() - t0
            speed = done / max(el, 0.1)
            eta = (len(need) - done) / speed / 60
            print(f'  {done}/{len(need)} (成功{ok} 失败{fail}) 速度{speed:.1f}张/s ETA {eta:.1f}分')

print(f'\n完成: 成功{ok} 失败{fail} 耗时{(time.time()-t0)/60:.1f}分, 目录共 {len(os.listdir(OUT_DIR))} 张')

# 更新清单 (4.4星+, 含预测)
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
