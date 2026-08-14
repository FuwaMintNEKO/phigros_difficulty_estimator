# -*- coding: utf-8 -*-
"""解压 Retribution 完整版并预测 (官谱格式, 24线)"""
import zipfile, os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
import app
from feature_extractor import extract_features

DL = 'C:/Users/NaNK/Downloads'
zpath = os.path.join(DL, 'Retribution ~Cycle of Redemption~.zip')
out = os.path.join(DL, 'Retribution_FULL.json')

if not os.path.exists(out):
    z = zipfile.ZipFile(zpath)
    z.extract('647836628.json', DL)
    os.rename(os.path.join(DL, '647836628.json'), out)
    print('解压完成:', out)
else:
    print('已存在:', out)

with open(out, 'rb') as f:
    raw = f.read()
cd, pe = load_chart_from_bytes(raw)
print('格式检测:', 'standard' if pe is None else 'pe')
feats = extract_features(cd)
print('特征 keys:', len(feats))
print('total_notes:', feats.get('total_notes'))
print('speed_event_count:', feats.get('speed_event_count'), 'speed_std:', feats.get('speed_std'),
      'speed_volatility:', feats.get('speed_volatility'))
print('jline_move:', feats.get('jline_movement_density'), 'rotate:', feats.get('jline_rotate_density'),
      'disappear:', feats.get('jline_disappear_density'))
print('hold_count:', feats.get('hold_count') if 'hold_count' in feats else 'n/a')

for L in ['EZ', 'HD', 'IN', 'AT']:
    res, err = app.predict_one_chart(cd, speed=1.0, level=L)
    if res:
        print(f'{L}: pred={res["prediction"]:.2f} (gb={res["gb"]:.2f} boost={res["boost"]:.2f}) '
              f'cat={ {k: round(v,2) for k,v in res["categories"].items()} }')
