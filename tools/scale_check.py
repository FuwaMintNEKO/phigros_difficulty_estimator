# -*- coding: utf-8 -*-
"""验证: 条件缩放(双指抬/多指压)对两个谱的影响 + jline缺失影响"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
def feats_of(path):
    with open(path, 'rb') as f:
        cd, raw = load_chart_from_bytes(f.read())
    return extract_features(cd, speed=1.0)

yumeka = feats_of(os.path.join(DL, '夢の降る日に', '5333883479687925.json'))   # 双指 mf3=0
der = feats_of(os.path.join(DL, 'Der Schneid(1)', '1903581575578621.json'))     # 多指 mf3=15

# 1) 无条件缩放预测 (手工复刻 compute_boost 但所有scale=1)
def boost_noscale(feats):
    total = 0.0
    cap = app_mod.CAPS.get('_default', None)
    for fname, bl, co in app_mod.MANUAL_FLAT:
        v = feats.get(fname, 0)
        pv = app_mod.P95.get(fname, 0)
        t = max(pv*0.55, bl*0.5)
        if v <= t: continue
        e = v/t - 1.0
        c = app_mod.CAPS.get(fname, cap)
        if c is not None and e > c: e = c
        x = co * (e**0.70)
        if v > max(app_mod.P99.get(fname, 0), bl*0.5):
            pe = v/max(app_mod.P99.get(fname,0), bl*0.5) - 1.0
            if c is not None and pe > c: pe = c
            x += co * max(0,pe)**0.70*0.5
        total += x
    return total

# GB预测 (IN_AT onehot)
def gb_pred(feats, lv='IN'):
    lv2 = 'IN_AT' if lv in ('IN','AT') else lv
    vec = [0.0]*len(app_mod.LV_ORDER); vec[app_mod.LV_ORDER.index(lv2)] = 1.0
    x = np.array([[feats.get(n,0) for n in app_mod.FN] + vec])
    return float(app_mod.gb.predict(app_mod.scaler.transform(x))[0])

for nm, feats, truth, lv in [('夢降日(双指)', yumeka, 16.6, 'IN'), ('DerSchneid(多指)', der, 17.5, 'AT')]:
    b_cur, _, _ = app_mod.compute_boost(dict(feats), 1.0, is_custom=True)
    b_ns = boost_noscale(feats)
    g = gb_pred(feats, lv)
    print(f'\n{nm}: 官谱定数={truth}')
    print(f'  GB={g:.3f} 当前boost={b_cur:.3f} 无缩放boost={b_ns:.3f} 缩放差异={b_ns-b_cur:+.3f}')
    print(f'  当前总={g+b_cur:.3f}  无缩放总={g+b_ns:.3f}')
# 缩放系数本身
for nm, feats in [('夢降日(双指)', yumeka), ('DerSchneid(多指)', der)]:
    mf3 = feats.get('multi_finger_3plus_events', 0)
    dens = feats.get('above_avg_density_mean', 0)
    print(f'{nm}: mf3={mf3} dens={dens:.1f} → 条件缩放档: mf3<=5 → eff抬升档')
print('DONE')