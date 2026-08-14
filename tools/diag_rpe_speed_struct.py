# -*- coding: utf-8 -*-
"""确认 RPE 谱 speedEvents 字段结构 (start/end vs value) 与值分布"""
import json, os

DL = r'C:\Users\NaNK\Downloads'
targets = [
    ('Sigma(v3愚人节)', os.path.join(DL, 'Sigma (Haocore Mix) ~ 105秒の伝說 ~.json')),
    ('Retri残缺(RPE)', os.path.join(DL, '51030697.json')),
    ('ボーカル', os.path.join(DL, 'ボーカルに無茶させんな.json')),
    ('ふたり', os.path.join(DL, 'ふたりのスタートボタン(13.4).json')),
]

def fmt_time(t):
    if isinstance(t, list):
        return str(t)
    return str(t)

for name, path in targets:
    if not os.path.exists(path):
        print(f'--- {name}: 不存在'); continue
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        data = json.load(f)
    print(f'--- {name}: META.RPEVersion={data.get("META", {}).get("RPEVersion")}, '
          f'judgeLines={len(data.get("judgeLineList", []))}')
    for i, jl in enumerate(data.get('judgeLineList', [])[:3]):
        # 顶层
        top_se = jl.get('speedEvents', [])
        layers = jl.get('eventLayers', []) or []
        layer_se = []
        for layer in layers:
            if isinstance(layer, dict):
                layer_se.extend(layer.get('speedEvents', []))
        se = top_se or layer_se
        if se:
            print(f'  线{i}: speedEvents n={len(se)}, 前3个:')
            for ev in se[:3]:
                print('     ', json.dumps(ev, ensure_ascii=False)[:200])
        # notes 的 speed 字段
        notes = jl.get('notes', []) or jl.get('notesAbove', []) or jl.get('notesBelow', [])
        speeds = set()
        for n in notes[:500]:
            if 'speed' in n:
                speeds.add(n.get('speed'))
        print(f'  线{i}: notes={len(notes)}, 音符speed值集合(前10)={sorted(speeds)[:10]}')
