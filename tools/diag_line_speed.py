# -*- coding: utf-8 -*-
"""检查 Retribution / Sigma 的判定线级速度结构 (线bpm / speedEvents / 事件层)"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_speed_events

FILES = {
    'Retribution':     r'C:\Users\NaNK\Downloads\51030697.json',
    'Sigma (Regrets)': r'C:\Users\NaNK\Downloads\Sigma (Haocore Mix) ~ Regrets of The Yellow Tuli.json',
}

for k, p in FILES.items():
    with open(p, 'rb') as fh:
        raw = fh.read()
    cd, _ = load_chart_from_bytes(raw)
    jls = cd.get('judgeLineList', [])
    print(f'\n##### {k}: 判定线数 = {len(jls)} #####')
    for i, jl in enumerate(jls):
        bpm = jl.get('bpm')
        n_ab = len(jl.get('notesAbove', []))
        n_bl = len(jl.get('notesBelow', []))
        top_se = len(jl.get('speedEvents', []))
        layers = jl.get('eventLayers', [])
        layer_se = sum(len(l.get('speedEvents', [])) for l in layers)
        layer_cnt = len(layers)
        se_vals = [ev.get('value') for ev in jl.get('speedEvents', [])]
        ly_vals = [ev.get('value') for l in layers for ev in l.get('speedEvents', [])]
        # 其他事件计数
        mv = len(jl.get('judgeLineMoveEvents', []))
        rt = len(jl.get('judgeLineRotateEvents', []))
        dp = len(jl.get('judgeLineDisappearEvents', []))
        if n_ab + n_bl or bpm is not None or top_se or layer_se:
            print(f'  线{i}: bpm={bpm}  notes={n_ab}+{n_bl}  topSE={top_se}{se_vals[:3]}  '
                  f'layers={layer_cnt}  layerSE={layer_se}{ly_vals[:3]}  move={mv} rot={rt} dis={dp}')
