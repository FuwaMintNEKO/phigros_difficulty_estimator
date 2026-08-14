# -*- coding: utf-8 -*-
import json
off = json.load(open(r'data\chart\DerSchneid.Ωμεγα.0\AT.json', encoding='utf-8'))
jls = off['judgeLineList']
counts = sorted([len(jl.get('notesAbove', [])) + len(jl.get('notesBelow', [])) for jl in jls], reverse=True)
print('DerSchneid AT 每线音符数(降序):', counts[:12])
print('max线:', counts[0], ' 总:', sum(counts), ' 线数:', len(jls))

for song, lv in [('Glaciaxion.SunsetRay.0', 'IN'), ('Rrharil.TeamGrimoire.0', 'AT'),
                 ('GungnirFracture.Kryexe.0', 'IN'), ('夢の降る日に.seatrus.0', 'IN'),
                 ('Cthugha.USAO.0', 'AT'), ('Igallta.SeURa.0', 'AT')]:
    try:
        d = json.load(open(rf'data\chart\{song}\{lv}.json', encoding='utf-8'))
        c = sorted([len(jl.get('notesAbove', [])) + len(jl.get('notesBelow', [])) for jl in d['judgeLineList']], reverse=True)
        print(f'{song} {lv}: 线数={len(d["judgeLineList"])} max线={c[0]} 总={sum(c)}')
    except Exception as e:
        print(song, lv, 'ERR', e)
