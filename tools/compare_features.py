"""分析这个谱面和训练集中17+谱面的特征差异"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
import json
import numpy as np

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
NEW_CHART = os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json')

# 加载新谱面
with open(NEW_CHART, 'r', encoding='utf-8') as f:
    raw = json.load(f)

def to_beat_time(t):
    return t[0] * 4 + t[1] + (t[2] - 1) / 192.0

meta = raw.get('META', {})
print(f'=== 新谱面: {meta.get("name")} ===')
print(f'标注: {meta.get("level")}')
print(f'BPM: {raw.get("BPMList", [{}])[0].get("bpm")}')

# 统计note类型
taps = drags = holds = flicks = 0
for line in raw.get('judgeLineList', []):
    for n in line.get('notes', []):
        t = n.get('type', 0)
        if t == 1: taps += 1
        elif t == 2: drags += 1
        elif t == 3: holds += 1
        elif t == 4: flicks += 1
total = taps + drags + holds + flicks
print(f'  Tap={taps} Drag={drags} Hold={holds} Flick={flicks}')
print(f'  总={total}, Flick占比={flicks/total*100:.1f}%')

start = to_beat_time(raw['judgeLineList'][0].get('notes', [{}])[0].get('startTime', [0,0,1])) if raw['judgeLineList'][0].get('notes') else 0
# Find last note
last_time = 0
for line in raw.get('judgeLineList', []):
    for n in line.get('notes', []):
        t = to_beat_time(n.get('startTime', [0,0,1]))
        last_time = max(last_time, t)
duration_beats = last_time - start
bpm = raw.get('BPMList', [{}])[0].get('bpm', 120)
duration_sec = (duration_beats / bpm) * 1.875
print(f'  实际note时间跨度: {duration_beats:.0f} beats = {duration_sec:.1f}秒')
print(f'  密度: {total/max(duration_sec,0.01):.1f} notes/sec')

# 对比训练集中17+谱面的特征
song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)

high_diffs = []
for folder_name, info in chart_files.items():
    song_id = info['song_id']
    if song_id not in song_difficulties:
        continue
    diffs = song_difficulties[song_id]
    for level in ['EZ', 'HD', 'IN', 'AT']:
        if level in info['levels'] and level in diffs and diffs[level] >= 17.0:
            high_diffs.append((diffs[level], level, info['levels'][level], folder_name))

print(f'\n=== 训练集中 {len(high_diffs)} 个17+谱面特征对比 ===')

feats_17 = []
for d, lv, fp, name in high_diffs:
    try:
        cd = load_chart_json(fp)
        f = extract_features(cd)
        if f:
            feats_17.append((d, lv, name, f))
    except:
        pass

# 提取新谱面的特征
with open(NEW_CHART, 'r', encoding='utf-8') as f:
    new_chart_data = json.load(f)
from predict_rpe import convert_rpe_to_standard
converted = convert_rpe_to_standard(new_chart_data)
new_feats = extract_features(converted)

print(f'\n{"特征名":>30s} | {"新谱面":>8s} | {"17+平均":>8s} | {"17+范围":>16s}')
print('-'*68)
for fname in ['notes_per_second', 'tap_per_second', 'max_simultaneous', 
              'multi_finger_3plus_events', 'multi_finger_max_simultaneous',
              'mf_burst_count', 'wide_jump_count', 'flick_count', 'flick_ratio',
              'hold_lock_tap_events', 'hold_lock_avg_displacement',
              'core_micro_max_0.125beat', 'sustained_density_run_count',
              'track_section_count', 'density_transition_mean',
              'offbeat_ratio', 'rhythm_entropy', 'tempo_change_count']:
    nv = new_feats.get(fname, 0)
    vals = [f[3].get(fname, 0) for f in feats_17]
    avg = np.mean(vals) if vals else 0
    lo = min(vals) if vals else 0
    hi = max(vals) if vals else 0
    print(f'{fname:>30s} | {nv:>8.2f} | {avg:>8.2f} | {lo:>6.2f}~{hi:>6.2f}')
