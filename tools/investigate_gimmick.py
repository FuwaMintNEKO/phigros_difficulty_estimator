# -*- coding: utf-8 -*-
"""调查 gimmick 谱:
1. Chart_SP #1347 的 BPMList / 变速结构 (tempo_change=1298 是否解析异常)
2. スタートリップ vs 官方8级谱特征对比
"""
import sys, os, json
sys.path.insert(0, r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator')
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json

DL = r'C:\Users\NaNK\Downloads'

# ===== 1. Chart_SP BPM 结构 =====
p = os.path.join(DL, 'Chart_SP #1347(1).json')
d = json.load(open(p, encoding='utf-8'))
print('=== Chart_SP #1347 ===')
print('顶层keys:', list(d.keys()))
bl = d.get('BPMList', [])
print('BPM段数:', len(bl))
if bl:
    print('前5段BPM:', [(round(b['bpm'], 1), b['startTime']) for b in bl[:5]])
    # startTime 分布检查: 是否逐 tick 变化
    bpm_vals = [round(b['bpm'], 1) for b in bl]
    print('BPM去重后数量:', len(set(bpm_vals)))
    starts = [b['startTime'][0] for b in bl]
    print('BPM起始小节(min/max):', min(starts), max(starts))
jl = d.get('judgeLineList', [])
print('线数:', len(jl))
if jl:
    print('线0 keys:', list(jl[0].keys()))
    n0 = len(jl[0].get('notesAbove', [])) + len(jl[0].get('notesBelow', []))
    print('线0 音符数:', n0)

# ===== 2. スタートリップ vs 官方 8 级谱 =====
print()
print('=== スタートリップ vs 官方8级谱特征 ===')
print('(官方真定数 7.5-8.5 的谱面)')
p2 = os.path.join(DL, 'スタートリップ(12.2).json')
with open(p2, 'rb') as f:
    raw = f.read()
cd, pe = load_chart_from_bytes(raw)
f_startrip = extract_features(cd)

_ROOT = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator'
song_diffs = load_difficulty_tsv(os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv'))
cfs = find_chart_files(os.path.join(_ROOT, 'data', 'chart'))
items = []
for fn, info in cfs.items():
    sid = info['song_id']
    if sid not in song_diffs:
        continue
    diffs = song_diffs[sid]
    for lv in ['EZ', 'HD', 'IN', 'AT']:
        if lv in info['levels'] and lv in diffs and 7.5 <= diffs[lv] <= 8.5:
            items.append((fn, lv, diffs[lv], info['levels'][lv]))

# 采样几个8级谱
items = items[:6]
feats_8 = []
for fn, lv, d8, fp in items:
    try:
        c8 = load_chart_json(fp)
        f8 = extract_features(c8)
        if f8:
            feats_8.append((fn, lv, d8, f8))
    except Exception as e:
        print('  err', fn, str(e)[:50])

KEYS = ['total_notes', 'duration_sec', 'notes_per_second', 'real_core_notes_per_second',
        'real_active_sec', 'above_avg_density_mean', 'bpm', 'bpm_max', 'tempo_change_count',
        'hold_ratio', 'chord_ratio', 'position_entropy', 'jline_movement_density',
        'jline_rotate_density', 'jline_disappear_density', 'stair_density', 'trill_density']
print(f'{"特征":<28} {"スタートリップ":>12} {"官方8级均值":>12}')
for k in KEYS:
    vs = [f[k] for _, _, _, f in feats_8]
    avg = sum(vs) / len(vs) if vs else float('nan')
    print(f'{k:<28} {f_startrip.get(k, 0):>12.3f} {avg:>12.3f}')
print()
print('对比的8级谱:')
for fn, lv, d8, f8 in feats_8:
    print(f'  {fn} [{lv}] 定数{d8} notes={f8["total_notes"]} nps={f8["notes_per_second"]:.2f} tempo={f8["tempo_change_count"]}')
