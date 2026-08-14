# -*- coding: utf-8 -*-
"""批量下载 phira 谱面 pez, 只抽取其中的 .json 谱面保存

用法: python tools/fetch_phira_pez.py [--min-diff 16.5] [--max-count N]
从 data/phira/charts.json 选谱, 下载 file 链接, 解压出 {id}.json 存到 data/phira/json/
已存在则跳过, 支持断点续传
"""
import os, sys, json, time, io, zipfile
import urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUT_JSON = os.path.join(_ROOT, 'data', 'phira', 'json')
os.makedirs(OUT_JSON, exist_ok=True)
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--min-diff', type=float, default=0.0)
ap.add_argument('--max-count', type=int, default=99999)
args = ap.parse_args()


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


def main():
    charts = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
    allc = [c for lst in charts.values() for c in lst]
    # 去重 (按 id)
    seen, uniq = set(), []
    for c in allc:
        if c['id'] in seen:
            continue
        seen.add(c['id']); uniq.append(c)
    sel = [c for c in uniq if c['difficulty'] >= args.min_diff][:args.max_count]
    print(f'待下载 {len(sel)} 张 (min-diff={args.min_diff})')

    ok, fail = 0, []
    for i, c in enumerate(sel):
        out_path = os.path.join(OUT_JSON, f"{c['id']}.json")
        if os.path.exists(out_path):
            ok += 1
            continue
        try:
            raw = download(c['file'])
            if len(raw) < 100:
                raise ValueError('too small')
            z = zipfile.ZipFile(__import__('io').BytesIO(raw))
            # 谱面文件: 优先 .json (RPE), 其次 .pec (PE文本), 排除 info.txt 等非谱面文件
            cands = []
            for n in z.namelist():
                low = n.lower()
                if low.endswith('.json'):
                    cands.append((0, n != f"{c['id']}.json", n))
                elif low.endswith('.pec') and 'info' not in low:
                    cands.append((1, 0, n))
            if not cands:
                raise ValueError('no chart file inside')
            cands.sort(key=lambda t: (t[0], t[1], -z.getinfo(t[2]).file_size))
            with open(out_path, 'wb') as f:
                f.write(z.read(cands[0][2]))
            ok += 1
            if (i + 1) % 10 == 0:
                print(f'  [{i+1}/{len(sel)}] 完成 {ok}, 失败 {len(fail)}')
        except Exception as e:
            fail.append((c['id'], c['name'], str(e)[:60]))
        time.sleep(0.2)
    print(f'完成: 成功 {ok}, 失败 {len(fail)}')
    for cid, name, err in fail[:20]:
        print(f'  FAIL {cid} {name[:30]}: {err}')


if __name__ == '__main__':
    main()
