# -*- coding: utf-8 -*-
"""统计: 训练集中有多少官谱会被 _is_rpe_v3 误判 (单线>800音符且带事件)"""
import os, json
CHART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'chart')

misclass = []
ratio_high = []  # max_line_ratio >= 0.95 (真正单线谱特征)
total = 0
for folder in os.listdir(CHART_DIR):
    d = os.path.join(CHART_DIR, folder)
    if not os.path.isdir(d):
        continue
    for fname in os.listdir(d):
        if not fname.endswith('.json'):
            continue
        fp = os.path.join(d, fname)
        try:
            with open(fp, encoding='utf-8') as f:
                data = json.load(f)
            jls = data.get('judgeLineList', [])
            if not jls:
                continue
            total += 1
            max_n = 0
            total_n = 0
            has_events = False
            for jl in jls:
                n = len(jl.get('notesAbove', [])) + len(jl.get('notesBelow', []))
                max_n = max(max_n, n)
                total_n += n
                if any(k in jl for k in ['judgeLineMoveEvents', 'judgeLineDisappearEvents', 'judgeLineRotateEvents']):
                    has_events = True
            ratio = max_n / total_n if total_n else 0
            if ratio >= 0.95:
                ratio_high.append((folder, fname, max_n, total_n, round(ratio, 3)))
            if max_n > 800 and has_events:
                misclass.append((folder, fname, max_n, total_n, round(ratio, 3)))
        except Exception:
            pass

print(f'总谱面: {total}')
print(f'\n=== 会被 _is_rpe_v3 误判 (>800音符+事件): {len(misclass)} ===')
for folder, fname, mx, tn, r in misclass:
    print(f'  {folder}/{fname:<4} max线={mx:<5} 总={tn:<5} ratio={r}')
print(f'\n=== max_line_ratio >= 0.95 (接近真正单线谱): {len(ratio_high)} ===')
for folder, fname, mx, tn, r in ratio_high[:30]:
    print(f'  {folder}/{fname:<4} max线={mx:<5} 总={tn:<5} ratio={r}')
