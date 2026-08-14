# -*- coding: utf-8 -*-
"""诊断几个异常谱面的解析结果"""
import sys, os
sys.path.insert(0, r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator')
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

DL = r'C:\Users\NaNK\Downloads'
for fn in ['Chart_SP #1347(1).json', 'Chart_SP.json', 'スタートリップ(12.2).json', 'RENDA JOCEKY.json', 'Sigma (Haocore Mix) ~ 105秒の伝說 ~.json']:
    p = os.path.join(DL, fn)
    try:
        with open(p, 'rb') as f:
            raw = f.read()
        cd, pe = load_chart_from_bytes(raw)
        if not cd:
            print(f'{fn[:44]:<46} 解析失败')
            continue
        feats = extract_features(cd)
        lines = len(cd.get('judgeLineList', []))
        n_notes = sum(len(l.get('notesAbove', [])) + len(l.get('notesBelow', []))
                      for l in cd.get('judgeLineList', []))
        fmt = 'PE' if pe else ('RPE' if 'META' in cd else 'STD')
        print(f'{fn[:44]:<46} fmt={fmt} lines={lines} notes={n_notes}')
        print(f'    dur={feats.get("duration_sec"):.1f} rcnps={feats.get("real_core_notes_per_second"):.2f} '
              f'nps={feats.get("notes_per_second"):.2f} bpm={feats.get("bpm")}')
        print(f'    mv={feats.get("jline_movement_density")} rot={feats.get("jline_rotate_density")} '
              f'disc={feats.get("jline_disappear_density")} tempo={feats.get("tempo_change_count")}')
    except Exception as e:
        print(f'{fn[:44]:<46} ERROR: {str(e)[:80]}')
