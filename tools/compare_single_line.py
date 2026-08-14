# -*- coding: utf-8 -*-
"""对比: 真正RPE愚人节谱 (test_datas) vs 官方单线谱 (风屿 IN) 的结构差异"""
import json, os

def describe(path, label):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    print(f'=== {label} ===')
    print('  顶层keys:', list(data.keys()))
    print('  META:', 'META' in data)
    if 'META' in data:
        print('  META keys:', list(data.get('META', {}).keys()))
        print('  RPEVersion:', data.get('META', {}).get('RPEVersion'))
    jls = data.get('judgeLineList', [])
    print(f'  判定线数: {len(jls)}')
    # 主线信息
    main = max(jls, key=lambda j: len(j.get('notesAbove', [])) + len(j.get('notesBelow', [])))
    notes = main.get('notesAbove', []) + main.get('notesBelow', [])
    px = [n.get('positionX', 0) for n in notes]
    times = [n.get('time', 0) for n in notes]
    print(f'  主线音符: {len(notes)}  positionX范围=[{min(px):.1f},{max(px):.1f}] 唯一px数={len(set(round(x,2) for x in px))}')
    print(f'  主线事件: move={len(main.get("judgeLineMoveEvents", []))} rotate={len(main.get("judgeLineRotateEvents", []))} disappear={len(main.get("judgeLineDisappearEvents", []))} speed={len(main.get("speedEvents", []))}')
    # 每条线的音符数分布
    counts = sorted([len(j.get('notesAbove', [])) + len(j.get('notesBelow', [])) for j in jls], reverse=True)
    print(f'  各线音符数(降序前10): {counts[:10]}')
    print()

CHART = r'data\chart'
describe(os.path.join(CHART, '风屿.闫东炜.0', 'IN.json'), '官方单线谱: 风屿 IN (1156全主线)')
describe(os.path.join(CHART, 'test_datas', 'Chart_SP.json'), 'RPE愚人节: Chart_SP')
describe(os.path.join(CHART, 'test_datas', 'Chart_SP #1347(1).json'), 'RPE愚人节: Chart_SP #1347')
