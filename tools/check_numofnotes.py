# -*- coding: utf-8 -*-
"""验证 numOfNotes 是否能干净区分 RPE愚人节导出与官方谱"""
import json, os
CHART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'chart')

n_num = 0
n_no = 0
num_files = []
total = 0
for folder in os.listdir(CHART_DIR):
    d = os.path.join(CHART_DIR, folder)
    if not os.path.isdir(d):
        continue
    if folder in ('test_datas', 'used_test_data'):
        continue
    for fname in os.listdir(d):
        if not fname.endswith('.json'):
            continue
        fp = os.path.join(d, fname)
        try:
            with open(fp, encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict) or 'judgeLineList' not in data:
                continue
            total += 1
            if 'numOfNotes' in data:
                n_num += 1
                num_files.append((folder, fname))
            else:
                n_no += 1
        except Exception:
            pass

print(f'官方谱面总数: {total}')
print(f'含 numOfNotes 字段: {n_num}')
print(f'不含: {n_no}')
for f in num_files[:20]:
    print('  ', f)
