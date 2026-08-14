# -*- coding: utf-8 -*-
"""1) 验证 6 首新官谱能解析并提取特征
   2) 用 .pez (RPE转换) 对照官方 JSON: 同谱面两格式的特征差异 -> 校验 RPE 解析器
"""
import os, sys, zipfile, io, json
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from unified_parser import load_chart_from_bytes

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
NEW_SONGS = ['70MinutesFighters.かたぎり', 'Cleyera.Riya', 'DerSchneid.Ωμεγα',
             'GungnirFracture.Kryexe', '夢の降る日に.seatrus', '星拂云锦featkoi.S9ryne']

print('='*70)
print('1. 新官谱解析验证')
for folder in NEW_SONGS:
    d = os.path.join(CHART_DIR, folder + '.0')
    if not os.path.isdir(d):
        print(f'  !! 缺目录 {folder}')
        continue
    for fname in sorted(os.listdir(d)):
        fp = os.path.join(d, fname)
        try:
            cd = load_chart_json(fp)
            fe = extract_features(cd)
            if fe:
                print(f'  {folder}/{fname:<5} notes={fe["total_notes"]:<5} '
                      f'dur={fe["duration_sec"]:.0f}s nps={fe["notes_per_second"]:.2f} '
                      f'rcnps={fe["real_core_notes_per_second"]:.2f}')
            else:
                print(f'  !! {folder}/{fname} 特征为空')
        except Exception as e:
            print(f'  !! {folder}/{fname} 解析失败: {e}')

print('\n' + '='*70)
print('2. 官方JSON vs .pez(RPE) 特征对比 (校验RPE解析器)')

def load_pez(path):
    with open(path, 'rb') as f:
        raw = f.read()
    if raw[:2] == b'PK':  # zip
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = [n for n in z.namelist() if n.endswith('.json')]
            if not names:
                return None
            return load_chart_from_bytes(z.read(names[0]))
    return load_chart_from_bytes(raw)

PEZ_DIR = r'D:\迅雷下载\Phigros_Resource-master\Phigros_Resource-master\phira'
KEY_FEATS = ['total_notes', 'duration_sec', 'notes_per_second', 'real_core_notes_per_second',
             'core_peak_density_1sec_top5avg', 'above_avg_density_mean', 'density_dimension',
             'judge_line_count', 'jline_movement_density', 'jline_rotate_density',
             'jline_disappear_density', 'speed_event_count', 'rhythm_entropy',
             'tempo_change_count', 'stair_speed_avg', 'trill_density', 'jack_density',
             'weighted_mf_score_per_sec', 'finger_peak_tps', 'pattern_switch_rate',
             'tap_ratio', 'hold_ratio', 'drag_ratio', 'flick_ratio',
             'above_below_cross', 'note_clutter_ratio']

pairs = [('DerSchneid.Ωμεγα.0', 'AT', 'DerSchneid.Ωμεγα-AT.pez'),
         ('GungnirFracture.Kryexe.0', 'IN', 'GungnirFracture.Kryexe-IN.pez'),
         ('夢の降る日に.seatrus.0', 'IN', '夢の降る日に.seatrus-IN.pez')]

for folder, lv, pez_name in pairs:
    official = extract_features(load_chart_json(os.path.join(CHART_DIR, folder, lv + '.json')))
    # 找 .pez
    pez_path = None
    for sub in ['AT', 'IN', 'HD', 'EZ']:
        cand = os.path.join(PEZ_DIR, sub, pez_name)
        if os.path.exists(cand):
            pez_path = cand
            break
    if pez_path is None:
        print(f'  !! 未找到 {pez_name}')
        continue
    rpe_data, _ = load_pez(pez_path)
    if rpe_data is None:
        print(f'  !! {pez_name} 解析失败')
        continue
    rpe_fe = extract_features(rpe_data)
    if rpe_fe is None:
        print(f'  !! {pez_name} 特征为空')
        continue
    print(f'\n  === {folder} [{lv}] 官方 vs {pez_name} ===')
    n_diff = 0
    for k in KEY_FEATS:
        a = official.get(k, 0); b = rpe_fe.get(k, 0)
        diff = abs(a - b)
        rel = diff / max(abs(a), 0.001)
        flag = ''
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if rel > 0.15 and diff > 0.5:
                flag = '  <<<<<< 大差异'
                n_diff += 1
        print(f'    {k:<38} 官方={a:>10.3f}  RPE={b:>10.3f}  差={diff:>8.2f}{flag}')
    print(f'  大差异特征数: {n_diff}')
