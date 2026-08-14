# -*- coding: utf-8 -*-
"""jline差异根源: 高仿RPE vs 官谱standard 的判定线事件提取"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
# 高仿 RPE
p1 = os.path.join(DL, '夢の降る日に', '5333883479687925.json')
with open(p1, 'rb') as f:
    cd1, raw1 = load_chart_from_bytes(f.read())
# 官谱
p2 = os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json')
with open(p2, 'rb') as f:
    cd2, raw2 = load_chart_from_bytes(f.read())

for nm, cd in [('高仿RPE', cd1), ('官谱', cd2)]:
    if isinstance(cd, dict):
        jls = cd.get('judgeLineList', [])
        n_move = n_rotate = n_disappear = 0
        for jl in jls:
            n_move += len(jl.get('judgeLineMoveEvents', []))
            n_rotate += len(jl.get('judgeLineRotateEvents', []))
            n_disappear += len(jl.get('judgeLineDisappearEvents', []))
        print(f'{nm}: lines={len(jls)} move={n_move} rotate={n_rotate} disappear={n_disappear}')
        # 看一个move事件样例
        for jl in jls:
            if jl.get('judgeLineMoveEvents'):
                print('  move样例:', jl['judgeLineMoveEvents'][0])
                break
    else:
        print(f'{nm}: cd是list len={len(cd)}')
        n_move = n_rotate = n_disappear = 0
        for jl in cd:
            n_move += len(jl.get('judgeLineMoveEvents', []))
            n_rotate += len(jl.get('judgeLineRotateEvents', []))
            n_disappear += len(jl.get('judgeLineDisappearEvents', []))
        print(f'  lines={len(cd)} move={n_move} rotate={n_rotate} disappear={n_disappear}')
print('DONE')