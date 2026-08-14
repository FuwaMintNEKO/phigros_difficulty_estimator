# -*- coding: utf-8 -*-
"""完整管线对比: 官谱原谱 vs 高仿 (含校准)"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
from unified_parser import load_chart_from_bytes

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
cases = [
    ('夢降日 官谱IN (双指)', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json'), 'IN', False, 16.6),
    ('夢降日 高仿IN (双指)', os.path.join(DL, '夢の降る日に', '5333883479687925.json'), 'IN', True, 16.6),
    ('DerSchneid 官谱AT (多指)', os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json'), 'AT', False, 17.5),
    ('DerSchneid 高仿AT (多指)', os.path.join(DL, 'Der Schneid(1)', '1903581575578621.json'), 'AT', True, 17.5),
]
print(f'{"谱":<32}{"官谱定数":>8}{"预测":>7}{"gb":>7}{"boost":>7}{"误差":>7}')
for nm, p, lv, isc, truth in cases:
    with open(p, 'rb') as f:
        cd, raw = load_chart_from_bytes(f.read())
    r, err = app_mod.predict_one_chart(cd, speed=1.0, level=lv, is_custom=isc, chart_name=os.path.basename(p))
    if r:
        print(f'{nm:<32}{truth:>8.1f}{r["prediction"]:>7.2f}{r["gb"]:>7.2f}{r["boost"]:>7.2f}{r["prediction"]-truth:>+7.2f}')
        print(f'    tags={r["tags"]}')
    else:
        print(f'{nm}: {err}')
print('DONE')