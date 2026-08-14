# -*- coding: utf-8 -*-
"""诊断: Retri 在 cap4 下每个 MANUAL_FLAT 特征的贡献明细 (找出过度放大的变速特征)"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
from boost_config import MANUAL_FLAT

DL = 'C:/Users/NaNK/Downloads'
with open(os.path.join(DL, 'Retribution_FULL.json'), 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
fe = extract_features(cd)

CAP = 4.0
print(f'{"特征":<32} {"值":>10} {"bl":>6} {"co":>7} {"t":>8} {"excess":>8} {"贡献":>8}')
print('-' * 90)
rows = []
for fname, bl, co in MANUAL_FLAT:
    v = fe.get(fname, 0)
    pv = 0.0  # 官方谱 P95 对新特征≈0, 旧特征此处仅参考用 bl*0.5
    t = max(pv * 0.55, bl * 0.5)
    if v <= t:
        continue
    e = min(v / t - 1.0, CAP)
    x = co * (e ** 0.70)
    rows.append((x, fname, v, bl, co, t, e))
rows.sort(key=lambda r: -r[0])
for x, fname, v, bl, co, t, e in rows:
    print(f'{fname:<32} {v:>10.3f} {bl:>6.1f} {co:>7.3f} {t:>8.3f} {e:>8.3f} {x:>8.4f}')
print('-' * 90)
print(f'变速相关合计: {sum(x for x, f, *_ in rows if "speed" in f or "flash" in f):.3f}')
print(f'和弦重键合计: {sum(x for x, f, *_ in rows if "chord_jack" in f):.3f}')
print(f'jack合计: {sum(x for x, f, *_ in rows if "jack" in f):.3f}')
print(f'新特征合计: {sum(x for x, f, *_ in rows if any(k in f for k in ["note_speed", "chord_jack", "flash", "jack_"])):.3f}')
