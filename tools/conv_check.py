# -*- coding: utf-8 -*-
"""验证: convert_rpe_to_standard 后 eventLayers 是否保留"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from predict_rpe import convert_rpe_to_standard

# 高仿夢降日
p = os.path.join(_ROOT, 'tools', '_tmp_dl_charts', '夢の降る日に', '5333883479687925.json')
with open(p, 'rb') as f:
    cd, raw = load_chart_from_bytes(f.read())
print('load_chart_from_bytes 返回 cd type:', type(cd))
if isinstance(cd, dict):
    print('cd keys:', list(cd.keys())[:10])
    jls = cd.get('judgeLineList', [])
    print('judgeLineList:', len(jls))
    if jls:
        print('线0 keys:', list(jls[0].keys())[:12])
        print('线0 eventLayers:', type(jls[0].get('eventLayers')).__name__, len(jls[0].get('eventLayers', [])) if isinstance(jls[0].get('eventLayers'), list) else '无')
        for layer in (jls[0].get('eventLayers') or [])[:1]:
            if layer:
                print('  layer keys:', list(layer.keys()))
else:
    print('cd 是 list len:', len(cd))
    if cd:
        print('线0 keys:', list(cd[0].keys())[:12])
# 手动转
raw2 = json.load(open(p, encoding='utf-8'))
conv = convert_rpe_to_standard(raw2)
jls2 = conv['judgeLineList']
print('\n手动 convert 后 线0 eventLayers:', len(jls2[0].get('eventLayers', [])) if jls2 else '?')
print('DONE')