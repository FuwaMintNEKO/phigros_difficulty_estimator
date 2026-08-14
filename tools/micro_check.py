# -*- coding: utf-8 -*-
"""fast_ms 特征在 MANUAL_FLAT/FN 中的存在性"""
import os, sys, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
flat_names = [f for f, _, _ in app_mod.MANUAL_FLAT]
print('MANUAL_FLAT 中 fast_ms:', [f for f in flat_names if 'fast_ms' in f])
print('FN 中 fast_ms:', [f for f in app_mod.FN if 'fast_ms' in f])
print('P95 中 fast_ms:', [f for f in app_mod.P95 if 'fast_ms' in f])
# Melodiniq 的 micro 特征
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
feats = extract_features(cd, speed=1.0)
print('\nMelodiniq micro特征:')
for k in sorted(feats.keys()):
    if 'micro' in k or 'fast_ms' in k or 'burst' in k:
        print(f'  {k} = {feats[k]:.3f}')
# Verrückt
p2 = os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')
with open(p2, 'rb') as f:
    cd2, _ = load_chart_from_bytes(f.read())
f2 = extract_features(cd2, speed=1.0)
print('\nVerrückt IN micro特征:')
for k in sorted(f2.keys()):
    if 'micro' in k or 'fast_ms' in k or 'burst' in k:
        print(f'  {k} = {f2[k]:.3f}')
print('DONE')