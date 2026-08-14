# -*- coding: utf-8 -*-
"""重新精确计算: 高仿夢降日 vs 官谱 的 jline 特征"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
def feats_of(path):
    with open(path, 'rb') as f:
        cd, raw = load_chart_from_bytes(f.read())
    return extract_features(cd, speed=1.0)

f1 = feats_of(os.path.join(DL, '夢の降る日に', '5333883479687925.json'))
f2 = feats_of(os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json'))
for k in ['jline_movement_density', 'jline_rotate_density', 'jline_disappear_density', 'jline_relative_cross', 'above_below_cross', 'multi_line_sim_events']:
    print(f'{k:<30} 高仿={f1.get(k,0):.2f}  官谱={f2.get(k,0):.2f}')
# 官谱 move 事件总数
with open(os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json'), 'rb') as f:
    cd2, _ = load_chart_from_bytes(f.read())
tot = {'move':0,'rot':0,'dis':0}
if isinstance(cd2, dict):
    for jl in cd2.get('judgeLineList', []):
        tot['move'] += len(jl.get('judgeLineMoveEvents', []))
        tot['rot'] += len(jl.get('judgeLineRotateEvents', []))
        tot['dis'] += len(jl.get('judgeLineDisappearEvents', []))
print('\n官谱事件:', tot)
# 高仿事件
with open(os.path.join(DL, '夢の降る日に', '5333883479687925.json'), 'rb') as f:
    cd1, _ = load_chart_from_bytes(f.read())
tot1 = {'move':0,'rot':0,'dis':0}
for jl in cd1.get('judgeLineList', []):
    for layer in jl.get('eventLayers', []) or []:
        if layer:
            tot1['move'] += len(layer.get('moveXEvents', [])) + len(layer.get('moveYEvents', []))
            tot1['rot'] += len(layer.get('rotateEvents', []))
            tot1['dis'] += len(layer.get('disappearEvents', []))
print('高仿事件(eventLayers):', tot1)
print('DONE')