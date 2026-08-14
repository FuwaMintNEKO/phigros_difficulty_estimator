# -*- coding: utf-8 -*-
"""修复后(fast_ms只算tap+hold) Melodiniq vs 官谱 新特征值"""
import os, sys, io, json, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

cases = [
    ('Melodiniq', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')),
    ('Verrückt(16.5)', os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')),
    ('夢降日(16.6)', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')),
    ('DerSchneid(17.5)', os.path.join(_ROOT, 'data', 'chart', 'DerSchneid.Ωμεγα.0', 'AT.json')),
]
print(f'{"谱":<16}{"fast050":>8}{"fast100":>8}{"miniburst":>10}{"globaljack":>11}')
for nm, p in cases:
    with open(p, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    f_ = extract_features(cd, speed=1.0)
    print(f'{nm:<16}{f_.get("fast_ms_050_ratio",0):>8.3f}{f_.get("fast_ms_100_ratio",0):>8.3f}{f_.get("miniburst_count",0):>10}{f_.get("global_jack_count",0):>11}')
print('DONE')