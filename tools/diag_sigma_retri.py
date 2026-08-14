# -*- coding: utf-8 -*-
"""对比 Sigma(Haocore Mix) ~ Regrets of The Yellow Tuli 与 Retribution ~ Cycle of Redemption
找出 sigma 预测偏高 / retribution 预测偏低的原因。"""
import os, sys, json
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
from boost_config import MANUAL_FLAT
import app

FILES = {
    'Sigma (Regrets)': r'C:\Users\NaNK\Downloads\Sigma (Haocore Mix) ~ Regrets of The Yellow Tuli.json',
    'Retribution':     r'C:\Users\NaNK\Downloads\51030697.json',
}

charts = {}
for k, p in FILES.items():
    with open(p, 'rb') as f:
        raw = f.read()
    cd, _ = load_chart_from_bytes(raw)
    charts[k] = cd

# 1) 预测分解
print('=' * 100)
for k, cd in charts.items():
    res, err = app.predict_one_chart(cd, speed=1.0, level='AT')
    print(f'\n##### {k}  (level=AT) #####')
    print(f"预测={res['prediction']:.2f}  GB={res['gb']:.2f}  boost={res['boost']:.2f}  "
          f"notes={res['total_notes']}  {res['duration_sec']}s  bpm={res['bpm']}")
    print(f"类别: {json.dumps(res['categories'], ensure_ascii=False)}")
    print(f"原始: {json.dumps(res['cat_raws'], ensure_ascii=False)}")
    print('boost 贡献 top15:')
    for kf in res['key_features']:
        print(f"  {kf['name']:<28} contrib={kf['contribution']:>6.3f}  value={kf['value']:>10.2f}  "
              f"threshold={kf['threshold']:>8.2f}  v/t={kf['excess']:>6.2f}")

# 2) 全 MANUAL_FLAT 特征值对比
print('\n' + '=' * 100)
print('全 boost 特征值对比 (AT):')
feats = {k: extract_features(cd, speed=1.0) for k, cd in charts.items()}
print(f'{"特征":<28} {"Sigma(Regrets)":>14} {"Retribution":>14}  差异说明')
print('-' * 100)
for fname, bl, co in MANUAL_FLAT:
    a = feats['Sigma (Regrets)'].get(fname, 0)
    b = feats['Retribution'].get(fname, 0)
    mark = ''
    if b > a * 1.5:
        mark = '<< Retribution 明显更高'
    elif a > b * 1.5:
        mark = '<< Sigma 明显更高'
    print(f'{fname:<28} {a:>14.3f} {b:>14.3f}  {mark}')

# 3) 基础统计
print('\n' + '=' * 100)
for k in charts:
    f = feats[k]
    print(f'\n### {k} 基础统计 ###')
    for key in ['total_notes', 'duration_sec', 'bpm', 'bpm_min', 'bpm_max', 'bpm_change_count',
                'real_core_notes_per_second', 'above_avg_density_mean', 'above_avg_duration_sec',
                'avg_chord_size_poly', 'weighted_mf_score_per_sec', 'chord_alternation_rate',
                'chord_size_entropy', 'multi_finger_3plus_events', 'discrete_mf_ratio',
                'stair_speed_avg', 'stair_chord_ratio', 'position_entropy', 'jline_movement_density']:
        print(f'  {key:<30} = {f.get(key, 0):.4f}')

# 4) jack / 变速 专项诊断 (用户补充: 重键 + 差速变速是关键难点)
print('\n' + '=' * 100)
print('jack / 变速专项对比:')
JACK_SPEED_KEYS = ['global_jack_count', 'same_line_jack_count', 'same_line_jack_ratio',
                   'short_jack_count', 'long_jack_count', 'jack_max_run',
                   'jack_event_count', 'jack_total_steps', 'jack_density',
                   'tempo_change_count', 'tempo_change_ratio',
                   'speed_event_count', 'speed_event_density', 'speed_std', 'speed_volatility']
print(f'{"特征":<26} {"Sigma(Regrets)":>14} {"Retribution":>14}')
for key in JACK_SPEED_KEYS:
    a = feats['Sigma (Regrets)'].get(key, 0)
    b = feats['Retribution'].get(key, 0)
    print(f'{key:<26} {a:>14.3f} {b:>14.3f}')

# 5) speedEvents 明细 (值分布: 差速程度)
print('\n' + '=' * 100)
for k, p in FILES.items():
    with open(p, 'rb') as fh:
        raw = fh.read()
    cd, _ = load_chart_from_bytes(raw)
    from feature_extractor import collect_speed_events
    sev = collect_speed_events(cd.get('judgeLineList', []))
    vals = [ev['value'] for ev in sev]
    if vals:
        import numpy as np
        v = np.array(vals)
        print(f'{k}: speedEvents n={len(v)}  min={v.min():.3f} max={v.max():.3f} '
              f'mean={v.mean():.3f} std={v.std():.3f}  |v-1|>0.5占比={(np.abs(v-1)>0.5).mean():.1%}')
    else:
        print(f'{k}: 无 speedEvents')
