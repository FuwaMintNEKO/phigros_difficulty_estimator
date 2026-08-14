# -*- coding: utf-8 -*-
"""诊断: 重训后 Retri/Sigma 的 gb + boost 分解"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import app
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

DL = r'C:\Users\NaNK\Downloads'
for name, fn in [('Retribution_FULL', 'Retribution_FULL.json'),
                 ('Sigma Regrets', 'Sigma (Haocore Mix) ~ Regrets of The Yellow Tuli.json')]:
    with open(os.path.join(DL, fn), 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    feats = extract_features(cd)
    res, _ = app.predict_one_chart(cd, speed=1.0, level='AT')
    print(f'=== {name} ===')
    print(f'pred={res["prediction"]}  gb={res["gb"]}  boost={res["boost"]}')
    print(f'categories: {res["categories"]}')
    print('top贡献:')
    for kf in res['key_features'][:12]:
        print(f'  {kf["name"]:<30} 贡献={kf["contribution"]:>7.3f} 值={kf["value"]:>10.2f} t={kf["threshold"]:>8.2f} excess={kf["excess"]:>8.2f}')
    print()
