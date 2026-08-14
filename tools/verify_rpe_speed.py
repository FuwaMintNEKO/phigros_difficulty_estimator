# -*- coding: utf-8 -*-
"""验证 RPE 谱 speedEvents 修复后变速特征不再为0"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

DL = r'C:\Users\NaNK\Downloads'
targets = [
    ('Retri残缺(RPE)', '51030697.json'),
    ('ボーカル(RPE)', 'ボーカルに無茶させんな.json'),
    ('ふたり(RPE)', 'ふたりのスタートボタン(13.4).json'),
    ('Sigma(官谱格式)', 'Sigma (Haocore Mix) ~ 105秒の伝說 ~.json'),
    ('Retri完整版', 'Retribution_FULL.json'),
]
print(f'{"谱面":<22} {"ev_count":>9} {"mean":>7} {"std":>7} {"max":>7} {"volatility":>11}')
for name, fn in targets:
    p = os.path.join(DL, fn)
    if not os.path.exists(p):
        print(f'{name:<22} 不存在'); continue
    with open(p, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    fe = extract_features(cd)
    print(f'{name:<22} {fe.get("speed_event_count",0):>9} {fe.get("speed_mean",0):>7.2f} '
          f'{fe.get("speed_std",0):>7.2f} {fe.get("speed_max",0):>7.2f} {fe.get("speed_volatility",0):>11.2f}')
