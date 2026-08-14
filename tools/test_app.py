"""测试 app.py 预测"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os, json, numpy as np
sys.path.insert(0, '.')
from app import predict_one_chart
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)

# 测试几个极端谱面
test_names = [
    'BANGINGSTRIKE','DESTRUCTION321','Nhelv','ReEndofaDream',
    'FlutterEcho','Credits','opia','CervelleConnexion',
    'DataErr0r','HorizonBlue','狂喜蘭舞','游园地',
]

results = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    for lv in ['IN', 'AT']:
        if lv not in info.get('levels', {}): continue
        if lv not in song_difficulties[sid]: continue
        for pat in test_names:
            if pat.lower() in fn.lower():
                cd = load_chart_json(info['levels'][lv])
                result, err = predict_one_chart(cd)
                if result:
                    result['true_diff'] = song_difficulties[sid][lv]
                    result['name'] = fn[:30]
                    results.append(result)
                break

print(f'{"Name":<30s} {"True":>6s} {"Pred":>8s} {"Err":>7s} {"Boost":>7s}')
print('-'*65)
for r in sorted(results, key=lambda x: -x['true_diff']):
    err = r['prediction'] - r['true_diff']
    print(f'{r["name"]:<30s} {r["true_diff"]:>6.1f} {r["prediction"]:>8.2f} {err:>+7.2f} {r["boost"]:>7.2f}')