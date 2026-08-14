# -*- coding: utf-8 -*-
"""格式审计：实测官谱/RPE/PE 的关键字段分布，验证解析器假设"""
import os, json, glob, sys

ROOT = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator'
CHART_DIR = os.path.join(ROOT, 'data', 'chart')
DL = r'C:\Users\NaNK\Downloads'
PHIEDIT = r'D:\PhiEdit\Resources'

def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_standard(path):
    """官谱：positionX 分布、type 分布"""
    d = load(path)
    xs, types = [], []
    for jl in d.get('judgeLineList', []):
        for n in jl.get('notesAbove', []) + jl.get('notesBelow', []):
            xs.append(n.get('positionX', 0))
            types.append(n.get('type', 0))
    if not xs:
        return None
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    return {
        'notes': n,
        'x_min': xs_sorted[0], 'x_max': xs_sorted[-1],
        'x_p1': xs_sorted[int(n*0.005)], 'x_p99': xs_sorted[int(n*0.995)],
        'x_p5': xs_sorted[int(n*0.05)], 'x_p95': xs_sorted[int(n*0.95)],
        'types': {t: types.count(t) for t in set(types)},
    }

def analyze_rpe(path):
    """RPE：above 分布、type 分布、positionX 范围、startTime 范围"""
    d = load(path)
    metas = []
    above_cnt, types = {}, {}
    xs = []
    max_beat = 0
    for jl in d.get('judgeLineList', []):
        for n in jl.get('notes', []):
            a = n.get('above', None)
            above_cnt[a] = above_cnt.get(a, 0) + 1
            t = n.get('type', 0)
            types[t] = types.get(t, 0) + 1
            xs.append(n.get('positionX', 0))
            st = n.get('startTime', [0,0,1])
            if isinstance(st, list) and len(st) >= 2:
                beat = st[0] + st[1]/max(st[2],1)
                max_beat = max(max_beat, beat)
    dur = d.get('META', {}).get('duration', None)
    return {
        'name': d.get('META', {}).get('name', '?'),
        'notes': sum(above_cnt.values()),
        'above': above_cnt,
        'types': types,
        'x_min': min(xs) if xs else None, 'x_max': max(xs) if xs else None,
        'x_abs_max': max(abs(min(xs)), abs(max(xs))) if xs else None,
        'max_beat': max_beat,
        'META_duration': dur,
        'BPM_first': d.get('BPMList', [{}])[0].get('bpm'),
    }

def analyze_pe(path):
    """PE：n1/n2/n3/n4/bp/cp 时间范围"""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    stats = {'n1': [], 'n2': [], 'n3': [], 'n4': [], 'bp': [], 'cp': []}
    for raw in lines:
        raw = raw.strip()
        if not raw or raw.startswith('#') or raw.startswith('&'):
            continue
        parts = raw.split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd in stats and len(parts) >= 3:
            stats[cmd].append(float(parts[2]))
    out = {}
    for k, v in stats.items():
        if v:
            out[k] = {'min': min(v), 'max': max(v), 'count': len(v)}
    return out

print('=' * 70)
print('【1】官谱 (data/chart) positionX 实测')
print('=' * 70)
std_files = glob.glob(os.path.join(CHART_DIR, '*', '*.json'))
for p in std_files[:5]:
    r = analyze_standard(p)
    if r:
        print(os.path.basename(os.path.dirname(p)), r)

print()
print('=' * 70)
print('【2】RPE 谱面 (Downloads) above/type/坐标 实测')
print('=' * 70)
rpe_files = []
for fn in os.listdir(DL):
    if fn.endswith('.json'):
        p = os.path.join(DL, fn)
        try:
            d = load(p)
        except Exception:
            continue
        if isinstance(d, dict) and 'META' in d and 'RPEVersion' in d.get('META', {}):
            rpe_files.append(p)
        elif isinstance(d, dict) and 'judgeLineList' in d and d.get('judgeLineList') and \
             isinstance(d['judgeLineList'][0], dict) and 'notes' in d['judgeLineList'][0]:
            rpe_files.append(p)

for p in rpe_files[:8]:
    r = analyze_rpe(p)
    print(os.path.basename(p))
    print('   ', r)

print()
print('=' * 70)
print('【3】PhiEdit 自制谱 (Resources)')
print('=' * 70)
for root in os.listdir(PHIEDIT):
    rdir = os.path.join(PHIEDIT, root)
    if os.path.isdir(rdir):
        for fn in os.listdir(rdir):
            if fn.endswith('.json') and not fn.startswith('AutoSave'):
                p = os.path.join(rdir, fn)
                try:
                    d = load(p)
                except Exception:
                    continue
                if isinstance(d, dict) and 'judgeLineList' in d:
                    r = analyze_rpe(p)
                    print(os.path.join(root, fn))
                    print('   ', r)

print()
print('=' * 70)
print('【4】PE 格式 (スタートリップ) 时间单位')
print('=' * 70)
pe_path = os.path.join(DL, 'スタートリップ(12.2).json')
if os.path.exists(pe_path):
    r = analyze_pe(pe_path)
    for k, v in r.items():
        print(f'  {k}: min={v["min"]} max={v["max"]} count={v["count"]}')
    # 检查 META/duration
    with open(pe_path, 'r', encoding='utf-8') as f:
        first_lines = [l.strip() for l in f.read().split('\n')[:5] if l.strip()]
    print('  前5行:', first_lines)
