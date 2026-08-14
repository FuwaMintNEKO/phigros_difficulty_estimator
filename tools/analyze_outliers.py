# -*- coding: utf-8 -*-
"""诊断 CV 极端误差谱面: 对比其关键特征与同真定数档位的中位数, 找出特征盲区

重点考察:
  1. 密度类 (notes_per_sec, real_core_notes_per_second, peak密度)
  2. 读谱类 (jline_movement/rotate/disappear_density, above_below_cross, speed_*)
  3. 键型类 (stair, trill, jack, chord, finger)
  4. 标签合理性 (EZ/HD/IN/AT 跨难度差异)
"""
import os, sys, pickle, numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)

# 加载全量特征 (与 cv_analysis 一致)
all_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties:
        continue
    diffs = song_difficulties[sid]
    for lv in ['EZ', 'HD', 'IN', 'AT']:
        if lv in info['levels'] and lv in diffs:
            all_items.append({'folder': fn, 'filepath': info['levels'][lv],
                              'difficulty': diffs[lv], 'level': lv})

feats_list, labels, levels_list, names_list = [], [], [], []
for item in all_items:
    try:
        cd = load_chart_json(item['filepath'])
        fe = extract_features(cd)
        if fe:
            feats_list.append(fe); labels.append(item['difficulty'])
            levels_list.append(item['level']); names_list.append(item['folder'])
    except Exception:
        pass
n = len(feats_list)
y = np.array(labels)

# 目标谱面
TARGETS = [
    ('JourneywithYou.Iris.0', 'HD'), ('Bloom.targ麻团qwq.0', 'HD'),
    ('宇宙残骸少女CosmicDustyGirl.辻原一郎.0', 'HD'),
    ('宇宙残骸少女CosmicDustyGirl.辻原一郎.0', 'EZ'),
    ('下一秒.BobHouSherlock卓智媛feat兰音Reine.0', 'HD'),
    ('Ramification.rareguyReina.0', 'HD'),
    ('ジングルベルJingleBell.A39沙包P.0', 'EZ'),
    ('QuantumHyperspace.D_AAN.0', 'EZ'),
    ('Bougainvillea.Jade.0', 'EZ'),
    ('Adastraperaspera.RabbitHouse.0', 'EZ'),
]

KEY_FEATS = ['total_notes', 'duration_sec', 'notes_per_second', 'real_core_notes_per_second',
             'core_peak_density_1sec_top5avg', 'core_peak_density_top5avg_1beat',
             'above_avg_density_mean', 'density_dimension',
             'judge_line_count', 'jline_movement_density', 'jline_rotate_density',
             'jline_disappear_density', 'above_below_cross', 'speed_event_density',
             'speed_volatility', 'rhythm_entropy', 'tempo_change_count',
             'stair_speed_avg', 'stair_complexity', 'trill_density', 'jack_density',
             'chord_alternation_rate', 'weighted_mf_score_per_sec',
             'finger_peak_tps', 'pattern_switch_rate',
             'drag_ratio', 'flick_ratio', 'hold_ratio', 'tap_ratio',
             'note_clutter_ratio', 'track_deviation_score', 'offbeat_ratio']

for folder, lv in TARGETS:
    idx = None
    for i, (nm, lv_i) in enumerate(zip(names_list, levels_list)):
        if nm == folder and lv_i == lv:
            idx = i
            break
    if idx is None:
        print(f'!! 未找到 {folder} {lv}')
        continue
    # 同档位参照: 真定数 ±1.0 内
    band = np.where(np.abs(y - y[idx]) <= 1.0)[0]
    print(f'\n{"="*78}')
    print(f'{folder} [{lv}] 真定数={y[idx]:.1f}  n={len(band)} (参照±1.0)')
    for k in KEY_FEATS:
        v = feats_list[idx].get(k, 0)
        if k in ('above_below_cross', 'offbeat_ratio'):
            med = np.median([feats_list[j].get(k, 0) for j in band])
            flag = ''
            if isinstance(v, (int, float)) and isinstance(med, (int, float)) and med != 0:
                ratio = v / med
                flag = '  <<<<< 异常' if ratio > 3 or ratio < 1/3 else ''
            print(f'  {k:<38} 本谱={v:>10.3f}  档位中位={med:>10.3f}{flag}')
        elif isinstance(v, (int, float)):
            med = np.median([feats_list[j].get(k, 0) for j in band])
            flag = ''
            if med != 0:
                ratio = v / med
                if ratio > 4 or ratio < 0.25:
                    flag = '  <<<<< 异常'
            print(f'  {k:<38} 本谱={v:>10.3f}  档位中位={med:>10.3f}{flag}')
