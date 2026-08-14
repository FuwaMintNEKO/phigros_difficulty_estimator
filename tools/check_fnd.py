# -*- coding: utf-8 -*-
"""GB 特征重要性 (GB特征是fn+level onehot, 与FN错位3)"""
import os, sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
gb = app_mod.gb
imps = gb.feature_importances_
names = app_mod.FN + ['LV_EZ', 'LV_HD', 'LV_IN_AT']
print(f'imps: {len(imps)}, names: {len(names)}')
idx = np.argsort(-imps)[:40]
for i in idx:
    print(f'  {names[i]:<38} {imps[i]:.4f}')
print('DONE')