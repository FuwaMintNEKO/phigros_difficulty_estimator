# -*- coding: utf-8 -*-
"""v11.9 vs v11.10 完整管线对比: MANUAL_FLAT/P95/P99/caps/校准"""
import os, sys, numpy as np, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod

with open(os.path.join(_ROOT, 'models', '6dim_model_v11_9.pkl'), 'rb') as f:
    m9 = pickle.load(f)

print('=== v11.9 MANUAL_FLAT ===')
for row in m9['MANUAL_FLAT']:
    print(' ', row)
print('\n=== v11.10 MANUAL_FLAT ===')
for row in app_mod.MANUAL_FLAT:
    print(' ', row)
print('\nfn9 中 thirtysecond 相关:', [f for f in m9['feature_names'] if 'thirtysecond' in f])
print('fn10 中 thirtysecond 相关:', [f for f in app_mod.FN if 'thirtysecond' in f])
print('v11.9 校准表?', 'calib' in m9, [k for k in m9.keys()])
print('app 校准表:', app_mod._CALIB_TABLE if hasattr(app_mod, '_CALIB_TABLE') else '?')
print('DONE')