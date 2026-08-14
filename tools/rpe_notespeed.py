# -*- coding: utf-8 -*-
"""RPE高仿音符 speed 字段检查"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
p = os.path.join(DL, '夢の降る日に', '5333883479687925.json')
raw = json.load(open(p, encoding='utf-8'))
speeds = []
for jl in raw.get('judgeLineList', []):
    for n in jl.get('notes', []):
        speeds.append(n.get('speed', 1.0))
speeds = np.array(speeds)
print(f'音符: {len(speeds)}, speed非1.0: {np.sum(speeds != 1.0)}')
non1 = speeds[speeds != 1.0]
if len(non1):
    print(f'非1.0值: {sorted(set(non1))[:10]}')
# 官谱的297个音符speed 与 speedEvents 的关系: 官谱把speedEvent分配到每个音符?
print('\n=== 理解: 官谱 per-note speed = speedEvents插值结果? ===')
print('官谱 speedEvents 228个, 音符speed非1.0 297个')
print('官谱渲染器: 音符落速 = 所在判定线的 speedEvents 插值')
print('RPE: 音符 speed 字段直接给出 (但高仿全1.0?)')
print('\n=== 关键: RPE高仿的音符speed全1.0但speedEvents有变速 → 特征note_speed无法反映RPE变速 ===')
print('DONE')