import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'gb_final_model.pkl')

with open(MODEL_PATH, 'rb') as f:
    m = pickle.load(f)
feature_names = m['feature_names']

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)

all_at = {}
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    if 'AT' in info['levels'] and 'AT' in diffs:
        try:
            cd = load_chart_json(info['levels']['AT'])
            feats = extract_features(cd)
            if feats:
                all_at[fn] = {'true': diffs['AT'], 'feats': feats}
        except: pass

# 找高密度谱面(notes_per_second>8)
high_density = {k: v for k, v in all_at.items() if v['feats'].get('notes_per_second',0) > 8}
print(f'高密度(>8 nps) AT谱面: {len(high_density)}个')

# 标记"顺手" vs "卡手"
# 看这些谱面的标注难度，加上Rrhar'il作为参考
smooth_ones = ['QZKago Requiem', 'BANGING STRIKE', 'ERABYECONNEC10N', 'DESTRUCTION 3,2,1',
               'Re: End of a Dream', 'Hydra', 'Distorted Fate', 'AbsoluTedisoRdeR',
               'Stardust RAY', 'DER Richter']
awkward_ones = ['Rrhar\'il', 'Igallta', '祈-我ら神祖', 'Indelible Scar',
                'Spasmodic', 'Cuvism', 'PRAGMATISM', 'SATELLITE']

# 按名字匹配
smooth_data = []
awkward_data = []
for name, data in high_density.items():
    matched_s = False
    matched_a = False
    for s in smooth_ones:
        if s.lower().replace(' ', '').replace('\'', '') in name.lower().replace(' ', '').replace('.', ''):
            matched_s = True; break
    for a in awkward_ones:
        if a.lower().replace(' ', '').replace('\'', '') in name.lower().replace(' ', '').replace('.', ''):
            matched_a = True; break
    if matched_s: smooth_data.append((name, data))
    elif matched_a: awkward_data.append((name, data))

print(f'\n顺手高密谱: {len(smooth_data)}')
for n, d in smooth_data:
    print(f'  {n}: 标注={d["true"]:.1f} nps={d["feats"].get("notes_per_second",0):.1f}')

print(f'\n卡手高密谱: {len(awkward_data)}')
for n, d in awkward_data:
    print(f'  {n}: 标注={d["true"]:.1f} nps={d["feats"].get("notes_per_second",0):.1f}')

# 对比两组之间的特征差异
key_features = [
    ('notes_per_second', '总密度'),
    ('tap_per_second', 'Tap密度'),
    ('tempo_change_count', '节拍变化次数'),
    ('tempo_change_ratio', '节拍变化比例'),
    ('rhythm_entropy', '节奏熵(越高越乱)'),
    ('rhythm_diversity', '节奏多样性'),
    ('dominant_rhythm_ratio', '主导节奏占比(越高越单一)'),
    ('density_transition_max', '密度突变最大值'),
    ('density_transition_mean', '密度突变均值'),
    ('density_transition_std', '密度突变量'),
    ('stop_go_count', '骤停-骤起次数'),
    ('cross_hand_event_count', '交叉手事件'),
    ('cross_hand_ratio', '交叉手比例'),
    ('position_entropy', '位置熵(越高越离散)'),
    ('position_std', '位置标准差'),
    ('wide_jump_count', '大跳次数'),
    ('wide_jump_density', '大跳密度'),
    ('hold_lock_tap_events', 'Hold锁手事件'),
    ('hold_lock_avg_displacement', 'Hold锁手平均位移'),
    ('hold_lock_displacement_per_sec', 'Hold锁手位移密度'),
    ('micro_max_0.0625beat', '微窗口爆发(1/16)'),
    ('core_micro_max_0.125beat', '核心微窗口(1/8)'),
    ('sustained_density_run_count', '耐力持续段'),
    ('multi_finger_3plus_events', '3指+多押'),
    ('multi_finger_4plus_events', '4指+多押'),
    ('mf_burst_count', '多押爆发段'),
    ('mf_events_per_second', '多押密度'),
    ('burst_intensity_mean', '爆发强度均值'),
    ('dense_mf_count', '密集多押数'),
    ('concurrent_hold_events', 'Hold重叠事件'),
    ('avg_concurrent_holds', '平均同时Hold数'),
    ('burst_window_count', '爆发窗口数'),
    ('track_section_count', '定轨段数'),
    ('speed_change_total_impact', '变速冲击'),
    ('speed_event_count', '变速事件数'),
    ('offbeat_ratio', '反拍比例'),
    ('weak_beat_ratio', '弱拍比例'),
    ('interval_cv', '间隔变异系数'),
    ('short_interval_ratio', '短间隔比例'),
    ('hand_speed_index', '手速指数'),
    ('tap_burst_peak_to_mean', 'Tap爆发峰均比'),
]

print(f'\n{"="*80}')
print(f'{"特征":36s} {"顺手组":>10s} {"卡手组":>10s} {"差值":>8s} {"方向"}')
print(f'{"-"*80}')

for fname, desc in key_features:
    svals = [d['feats'].get(fname, 0) for _, d in smooth_data]
    avals = [d['feats'].get(fname, 0) for _, d in awkward_data]
    sm = np.mean(svals) if svals else 0
    am = np.mean(avals) if avals else 0
    diff = sm - am
    direction = '顺手>卡手' if diff > 0 else '卡手>顺手' if diff < 0 else '相同'
    print(f'  {fname:36s} {sm:10.4f} {am:10.4f} {diff:+8.4f} {direction}')
