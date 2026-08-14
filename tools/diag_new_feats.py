# -*- coding: utf-8 -*-
"""诊断: Retribution完整版 vs Sigma 的新特征(jack/差速/闪现)实际值"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

DL = 'C:/Users/NaNK/Downloads'
targets = [
    ('Retribution_FULL', os.path.join(DL, 'Retribution_FULL.json')),
    ('Sigma', os.path.join(DL, 'Sigma (Haocore Mix) ~ 105秒の伝說 ~.json')),
    ('Gungnir(冈格尼尔)', os.path.join(DL, '0683632416398134.json')),  # 无此文件则跳过
]
NEW = ['jack_density', 'jack_max_run', 'same_line_jack_ratio', 'long_jack_count',
       'note_speed_non1_ratio', 'note_speed_std', 'note_speed_max', 'note_speed_density',
       'fast_hold_ratio', 'flash_hold_ratio', 'chord_jack_steps', 'chord_jack_density',
       'chord_jack_3plus_pairs']
JACK_RAW = ['global_jack_count', 'same_line_jack_count', 'short_jack_count', 'long_jack_count',
            'jack_event_count', 'jack_total_steps', 'miniburst_count', 'miniburst_density']

for name, path in targets:
    if not os.path.exists(path):
        print(f'--- {name}: 文件不存在, 跳过')
        continue
    with open(path, 'rb') as f:
        raw = f.read()
    cd, _ = load_chart_from_bytes(raw)
    fe = extract_features(cd)
    print(f'--- {name}: 总音符={fe.get("total_notes")}, 时长={fe.get("duration_sec"):.1f}s')
    for k in JACK_RAW + NEW:
        if k in fe:
            print(f'    {k:<28} = {fe[k]}')
    print()
