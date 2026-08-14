# -*- coding: utf-8 -*-
"""Bathin 7516.json 用 load_chart_from_bytes 解析后检查speed"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

p = os.path.join(_ROOT, 'data', 'phira', 'json', '7516.json')
with open(p, 'rb') as f:
    cd, raw = load_chart_from_bytes(f.read())
print('解析成功, 顶层keys:', list(cd.keys())[:25])
# 找notes数组
for k, v in cd.items():
    if isinstance(v, list) and v and isinstance(v[0], dict) and 'type' in v[0]:
        notes = v
        print('notes key:', k, 'len:', len(v))
        has_speed = sum(1 for n in v if 'speed' in n)
        print('有speed:', has_speed)
        speeds = sorted(set(float(n.get('speed', 1.0)) for n in v))
        print('speed集合:', speeds[:25])
        holds = [n for n in v if n.get('type') == 3]
        print('长条:', len(holds), '长条speed:', sorted(set(float(n.get('speed',1.0)) for n in holds))[:15])
        break
# 特征
feats = extract_features(cd, speed=1.0)
print('\nnote_speed_non1_ratio:', feats.get('note_speed_non1_ratio'), 'std:', feats.get('note_speed_std'), 'max:', feats.get('note_speed_max'))
print('fast_hold_ratio:', feats.get('fast_hold_ratio'), 'flash_hold_ratio:', feats.get('flash_hold_ratio'))
print('DONE')