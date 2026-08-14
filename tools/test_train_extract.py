# -*- coding: utf-8 -*-
"""快速测试: unranked训练集特征提取是否正常 (取5首)"""
import os, sys, io, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star')
with open(os.path.join(_ROOT, 'data', 'phira', 'train_unranked_1000.csv'), encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
print('训练集行数:', len(rows))
ok = fail = 0
for row in rows[:8]:
    p = os.path.join(JSON_DIR, row['id'] + '.json')
    try:
        with open(p, 'rb') as f:
            cd, raw = load_chart_from_bytes(f.read())
        feats = extract_features(cd, speed=1.0)
        if feats:
            ok += 1
            print(f"  OK {row['id']} {row['name'][:20]} diff={row['diff']} lv={row['level']} n_feats={len(feats)}")
        else:
            fail += 1; print(f"  FAIL(无特征) {row['id']} {row['name'][:20]}")
    except Exception as ex:
        fail += 1
        print(f"  ERR {row['id']}: {ex}")
print(f'\n测试: ok={ok} fail={fail}')
print('DONE')