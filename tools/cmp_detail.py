# -*- coding: utf-8 -*-
"""高仿 vs 官谱: 预测分解对比 (新v11.13)"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
from unified_parser import load_chart_from_bytes

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
cases = [
    ('夢降日 高仿IN', os.path.join(DL, '夢の降る日に', '5333883479687925.json'), 'IN'),
    ('夢降日 官谱IN', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json'), 'IN'),
    ('DerSchneid 高仿IN', os.path.join(DL, 'Der Schneid(1)', '1903581575578621.json'), 'IN'),
    ('DerSchneid 官谱AT', os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json'), 'AT'),
]
print(f'{"谱":<22}{"预测":>7}{"gb":>7}{"boost":>7}{"校准前":>8}')
for nm, p, lv in cases:
    with open(p, 'rb') as f:
        cd, raw = load_chart_from_bytes(f.read())
    r, err = app_mod.predict_one_chart(cd, speed=1.0, level=lv, is_custom=True, chart_name=os.path.basename(p))
    if r:
        print(f'{nm:<22}{r["prediction"]:>7.2f}{r["gb"]:>7.2f}{r["boost"]:>7.2f}{r["prediction"]+0.1:>8.2f}')
    else:
        print(f'{nm}: {err}')
print('DONE')