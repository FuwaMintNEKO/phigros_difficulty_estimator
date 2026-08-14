# -*- coding: utf-8 -*-
"""直接检查 Retribution 原始 JSON 的判定线结构 (不经过统一解析器)"""
import json

p = r'C:\Users\NaNK\Downloads\51030697.json'
with open(p, 'r', encoding='utf-8', errors='ignore') as f:
    data = json.load(f)

print('顶层 keys:', list(data.keys()))
jls = data.get('judgeLineList', [])
print(f'judgeLineList 线数: {len(jls)}')
if jls:
    print('第一条线 keys:', list(jls[0].keys()))
    for i, jl in enumerate(jls[:6]):
        print(f'\n--- 线{i} ---')
        print('  bpm:', jl.get('bpm'))
        for k in jl:
            v = jl[k]
            n = len(v) if isinstance(v, list) else None
            print(f'  {k}: {"list[%d]" % n if isinstance(v, list) else v}')
    print(f'\n... 共 {len(jls)} 条线')
    # 统计有音符的线
    n_with_notes = sum(1 for jl in jls if len(jl.get('notesAbove', [])) + len(jl.get('notesBelow', [])) > 0)
    n_with_events = sum(1 for jl in jls if len(jl.get('judgeLineMoveEvents', [])) + len(jl.get('judgeLineRotateEvents', [])) + len(jl.get('judgeLineDisappearEvents', [])) > 0)
    n_with_speed = sum(1 for jl in jls if len(jl.get('speedEvents', [])) > 0)
    print(f'有音符的线: {n_with_notes}, 有move/rot/dis事件的线: {n_with_events}, 有speedEvents的线: {n_with_speed}')
    # speedEvents 值分布
    all_speeds = [ev.get('value') for jl in jls for ev in jl.get('speedEvents', [])]
    print('speedEvents 值:', all_speeds[:30], f'...共{len(all_speeds)}')
