# -*- coding: utf-8 -*-
"""深入 Retribution 的 RPE 高级事件结构 (eventLayers/posControl/alphaControl/yControl/BPMList/multiLineString)"""
import json

p = r'C:\Users\NaNK\Downloads\51030697.json'
with open(p, 'r', encoding='utf-8', errors='ignore') as f:
    data = json.load(f)

print('BPMList:', json.dumps(data.get('BPMList', []), ensure_ascii=False)[:500])
print('judgeLineGroup:', json.dumps(data.get('judgeLineGroup'), ensure_ascii=False)[:300])
print('multiLineString:', json.dumps(data.get('multiLineString'), ensure_ascii=False)[:300])
print('multiScale:', json.dumps(data.get('multiScale'), ensure_ascii=False)[:200])
print('META:', json.dumps(data.get('META'), ensure_ascii=False)[:400])

for i, jl in enumerate(data['judgeLineList']):
    print(f'\n======== 线{i} ========')
    print('bpmfactor:', jl.get('bpmfactor'))
    print('eventLayers:')
    for li, layer in enumerate(jl.get('eventLayers', [])):
        print(f'  layer{li} keys:', list(layer.keys()))
        for k, v in layer.items():
            if isinstance(v, list):
                vals = [json.dumps(x, ensure_ascii=False) for x in v[:4]]
                print(f'    {k}: n={len(v)} 样例={vals}')
            else:
                print(f'    {k}: {v}')
    for k in ['posControl', 'alphaControl', 'sizeControl', 'skewControl', 'yControl']:
        v = jl.get(k, [])
        if v:
            vals = [json.dumps(x, ensure_ascii=False)[:120] for x in v[:4]]
            print(f'{k}: n={len(v)} 样例={vals}')
    print('extended:', json.dumps(jl.get('extended'), ensure_ascii=False)[:600])
    # notes 的字段
    notes = jl.get('notes', [])
    if notes:
        print('note[0] keys:', list(notes[0].keys()))
        print('note[0]:', json.dumps(notes[0], ensure_ascii=False)[:300])
