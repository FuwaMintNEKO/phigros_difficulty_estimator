# -*- coding: utf-8 -*-
"""v11.13: 两个高仿谱 vs 官谱定数 完整诊断"""
import os, sys, io, json
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
    return extract_features(cd, speed=1.0), cd

yumeka_f, yumeka_cd = feats_of(os.path.join(DL, '夢の降る日に', '5333883479687925.json'))
der_f, der_cd = feats_of(os.path.join(DL, 'Der Schneid(1)', '1903581575578621.json'))

# 完整预测 (含校准)
r1, _ = app_mod.predict_one_chart(yumeka_cd, speed=1.0, level='IN', is_custom=True, chart_name='yumeka')
r2, _ = app_mod.predict_one_chart(der_cd, speed=1.0, level='IN', is_custom=True, chart_name='der')

print('=== v11.13 高仿谱完整预测 ===')
print(f'夢降日(双指): 预测={r1["prediction"]} 官谱定数=16.6 差={r1["prediction"]-16.6:+.2f}')
print(f'  gb={r1["gb"]:.3f} boost={r1["boost"]:.3f}')
print(f'  key贡献: ', [(k['name'], round(k['contribution'],3)) for k in r1['key_features'][:6]])
print()
print(f'DerSchneid(多指): 预测={r2["prediction"]} 官谱AT=17.5 差={r2["prediction"]-17.5:+.2f}')
print(f'  gb={r2["gb"]:.3f} boost={r2["boost"]:.3f}')
print(f'  key贡献: ', [(k['name'], round(k['contribution'],3)) for k in r2['key_features'][:6]])
print()
# 差异根源: 为什么夢降日双指被预测 16.41 (低0.19)?
# 对比官谱原谱的预测
off_yumeka = os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')
with open(off_yumeka, 'rb') as f:
    cd_off, _ = load_chart_from_bytes(f.read())
r3, _ = app_mod.predict_one_chart(cd_off, speed=1.0, level='IN', is_custom=True, chart_name='yumeka_off')
print(f'夢降日 官谱原谱(同管线): 预测={r3["prediction"]} (高仿={r1["prediction"]}) 差={r1["prediction"]-r3["prediction"]:+.2f}')
print(f'  高仿gb={r1["gb"]:.3f} 官谱gb={r3["gb"]:.3f} | 高仿boost={r1["boost"]:.3f} 官谱boost={r3["boost"]:.3f}')
print('DONE')