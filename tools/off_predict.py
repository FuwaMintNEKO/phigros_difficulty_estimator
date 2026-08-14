# -*- coding: utf-8 -*-
"""官谱原谱预测对比: DerSchneid / 夢の降る日に (official, is_custom=False)"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

# 官谱: data/chart/夢の降る日に.seatrus.0 和 DerSchneid.Ωμεγα.0 目录
for folder in ['夢の降る日に.seatrus.0', 'DerSchneid.Ωμεγα.0']:
    d = os.path.join(_ROOT, 'data', 'chart', folder)
    print(f'\n=== {folder} ===')
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if f.endswith('.json'):
            with open(p, 'rb') as fh:
                cd, raw = load_chart_from_bytes(fh.read())
            # 官方谱: is_custom=False
            r, err = app_mod.predict_one_chart(cd, speed=1.0, level='AT', is_custom=False, chart_name=f)
            if r:
                print(f'  {f}: 预测={r["prediction"]} gb={r["gb"]} boost={r["boost"]} tags={r["tags"]}')
            else:
                print(f'  {f}: {err}')
        elif os.path.isfile(p):
            print(f'  [文件] {f}')
print('DONE')