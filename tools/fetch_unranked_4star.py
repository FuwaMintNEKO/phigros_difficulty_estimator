# -*- coding: utf-8 -*-
"""下载未上架 4星+(rating>=0.8) 谱面到 data/phira/json_unranked_4star/ 并更新清单
已有文件直接复制, 缺失的从 phira file URL 下载(pez→json)
"""
import os, sys, json, time, io, zipfile, csv, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

META = os.path.join(_ROOT, 'data', 'phira', 'unranked_final_download.json')
SRC_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked')
OUT_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star')
os.makedirs(OUT_DIR, exist_ok=True)

meta = json.load(open(META, encoding='utf-8'))
print(f'未上架元数据: {len(meta)} 张')

# 4星线: phira rating 0-1 标度, 4星 = 0.8
sel = [c for c in meta if (c.get('rating') or 0) >= 0.8]
print(f'4星+(rating>=0.8): {len(sel)} 张 (rating范围 {min(c["rating"] for c in sel):.3f}-{max(c["rating"] for c in sel):.3f})')

def download(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
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

copied = downloaded = failed = 0
rows = []
for c in sel:
    cid = c['id']
    src = os.path.join(SRC_DIR, f'{cid}.json')
    dst = os.path.join(OUT_DIR, f'{cid}.json')
    try:
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)
            copied += 1
        else:
            raw = download(c['file'])
            data = extract_json(raw, cid)
            with open(dst, 'wb') as f:
                f.write(data)
            downloaded += 1
        rows.append({'id': cid, 'name': c['name'], 'level': c['level'], 'difficulty': c.get('difficulty'),
                     'rating': round(c.get('rating', 0), 4), 'ratingCount': c.get('ratingCount', 0)})
        if (copied + downloaded) % 100 == 0:
            print(f'  进度: {copied+downloaded}/{len(sel)}')
    except Exception as e:
        failed += 1
        print(f'  失败 {cid} {c["name"]}: {e}')
    time.sleep(0.3)  # 防限流

print(f'\n复制: {copied}, 下载: {downloaded}, 失败: {failed}')

# 更新清单 (加入 v11.2 预测)
def read_pred(path):
    preds = {}
    if not os.path.exists(path): return preds
    with open(path, encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f)
        head = next(rd)
        idx = {h: i for i, h in enumerate(head)}
        for c in rd:
            if len(c) < len(head): continue
            try:
                preds[int(c[idx['id']])] = {h: c[idx[h]] for h in head}
            except Exception: pass
    return preds
pred_map = read_pred(os.path.join(_ROOT, 'data', 'phira', 'v112_unranked_predictions_v2.csv'))

out_csv = os.path.join(_ROOT, 'data', 'phira', 'unranked_4star_list.csv')
with open(out_csv, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'name', 'level', 'difficulty', 'rating', 'ratingCount', 'pred', 'err', 'mf3', 'dens', 'nps', 'notes'])
    for r in sorted(rows, key=lambda x: -x['rating']):
        p = pred_map.get(r['id'], {})
        w.writerow([r['id'], r['name'], r['level'], r['difficulty'], r['rating'], r['ratingCount'],
                    p.get('pred', ''), p.get('err', ''), p.get('mf3', ''), p.get('dens', ''), p.get('nps', ''), p.get('notes', '')])
print(f'清单已写入: {out_csv} ({len(rows)} 行)')
print('DONE')
