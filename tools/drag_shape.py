# -*- coding: utf-8 -*-
"""对比: 官谱 drag 形态 vs Melodiniq RPE type4 形态"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# 官谱 drag (type=2) 的 holdTime
def official_drag_holdtime(path):
    raw = json.load(open(path, encoding='utf-8'))
    hts = []
    for jl in raw.get('judgeLineList', []):
        for n in jl.get('notesAbove', []) + jl.get('notesBelow', []):
            if n.get('type') == 2:  # Drag
                hts.append(n.get('holdTime', 0))
    return np.array(hts)

def rpe_type4_enddiff(path):
    """RPE type4 的 endTime-startTime (拍)"""
    raw = json.load(open(path, encoding='utf-8'))
    ds = []
    for jl in raw.get('judgeLineList', []):
        for n in jl.get('notes', []):
            if n.get('type') == 4:
                st, et = n.get('startTime'), n.get('endTime')
                if isinstance(st, list) and isinstance(et, list) and len(st)>=3 and len(et)>=3:
                    s = st[0] + st[1]/max(st[2],1)
                    e = et[0] + et[1]/max(et[2],1)
                    ds.append(abs(e-s))
    return np.array(ds)

DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
# 官谱 drag holdTime (tick, 1拍=32tick)
for nm, p in [('Verrückt', os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')),
              ('夢降日', os.path.join(_ROOT, 'data', 'chart', '夢の降る日に.seatrus.0', 'IN.json'))]:
    h = official_drag_holdtime(p)
    print(f'{nm} 官谱drag holdTime(tick): n={len(h)} P50={np.percentile(h,50):.1f} max={h.max():.1f} 零持续={np.mean(h==0)*100:.0f}%')

# RPE type4
for nm, p in [('Melodiniq', os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')),
              ('夢降日高仿', os.path.join(DL, '夢の降る日に', '5333883479687925.json'))]:
    d = rpe_type4_enddiff(p)
    print(f'{nm} RPE type4 endTime差(拍): n={len(d)} P50={np.percentile(d,50):.3f} max={d.max():.3f} 零持续={np.mean(d==0)*100:.0f}%')
print('\n官谱drag holdTime P50≈0 (瞬时); Melodiniq type4 endTime差 P50=0 (也瞬时)')
print('→ 形态一致: RPE type4 = 官谱 Drag (瞬时黄键)')
print('DONE')