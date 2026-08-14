# -*- coding: utf-8 -*-
"""批量检查 Downloads json 的谱面结构: 线数/音符/长条/speedEvents值"""
import os, json, sys

DL = r'C:\Users\NaNK\Downloads'
print(f'{"文件":<40} {"线数":>4} {"notes":>6} {"hold":>5} {"topSE":>5} {"SE值样例":<28} {"BPMList":>7}')
for fn in sorted(os.listdir(DL)):
    if not fn.lower().endswith('.json'):
        continue
    p = os.path.join(DL, fn)
    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            txt = f.read()
        if not txt.strip().startswith('{'):
            print(f'{fn[:40]:<40} {"PE文本":>4}')
            continue
        data = json.loads(txt)
    except Exception as e:
        print(f'{fn[:40]:<40} 读失败 {str(e)[:20]}')
        continue
    jls = data.get('judgeLineList', [])
    n_notes = 0
    n_hold = 0
    se_vals = []
    bpm_n = len(data.get('BPMList', [])) or 'line'
    for jl in jls:
        notes = jl.get('notes', []) or jl.get('notesAbove', []) or []
        n_notes += len(notes)
        n_hold += sum(1 for n in notes if n.get('type') == 3)
        for ev in jl.get('speedEvents', []):
            v = ev.get('value', ev.get('start', '?'))
            se_vals.append(v)
        for layer in jl.get('eventLayers', []) or []:
            if not isinstance(layer, dict):
                continue
            for ev in layer.get('speedEvents', []) or []:
                v = ev.get('value', ev.get('start', '?'))
                se_vals.append(v)
    se_sample = str(se_vals[:5])[:28] if se_vals else '-'
    print(f'{fn[:40]:<40} {len(jls):>4} {n_notes:>6} {n_hold:>5} {len(se_vals):>5} {se_sample:<28} {bpm_n:>7}')
