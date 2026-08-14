# -*- coding: utf-8 -*-
"""复现用户IN档结果: 两个高仿谱都按IN档预测"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
from unified_parser import load_chart_from_bytes

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
cases = [
    ('夢降日 高仿 IN档', os.path.join(DL, '夢の降る日に', '5333883479687925.json'), 'IN'),
    ('DerSchneid 高仿 IN档', os.path.join(DL, 'Der Schneid(1)', '1903581575578621.json'), 'IN'),
    ('DerSchneid 高仿 AT档', os.path.join(DL, 'Der Schneid(1)', '1903581575578621.json'), 'AT'),
]
for nm, p, lv in cases:
    with open(p, 'rb') as f:
        cd, raw = load_chart_from_bytes(f.read())
    r, err = app_mod.predict_one_chart(cd, speed=1.0, level=lv, is_custom=True, chart_name=os.path.basename(p))
    if r:
        print(f'{nm:<24} 预测={r["prediction"]:.2f} gb={r["gb"]:.2f} boost={r["boost"]:.2f}')
    else:
        print(f'{nm}: {err}')
print('DONE')