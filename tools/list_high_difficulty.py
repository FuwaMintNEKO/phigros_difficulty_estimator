import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from train_model import DifficultyModel
import numpy as np

base_chart = os.path.join(_ROOT, 'data', 'chart')
tsv = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

diffs = load_difficulty_tsv(tsv)
charts = find_chart_files(base_chart)

model = DifficultyModel()
model.load('models/unified_model.pkl')

results = []
for folder, info in charts.items():
    sid = info['song_id']
    if sid not in diffs: continue
    for level in ['EZ', 'HD', 'IN', 'AT']:
        if level not in info['levels'] or level not in diffs[sid]: continue
        actual = diffs[sid][level]
        if actual <= 15.5: continue
        try:
            data = load_chart_json(info['levels'][level])
            feats = extract_features(data)
            if feats is None: continue
            X = np.array([[feats.get(n, 0) for n in model.feature_names]])
            pred = model.predict(X, 'ensemble')
            if pred is not None:
                results.append((folder.replace('.0', ''), level, actual, float(pred[0])))
        except: pass

results.sort(key=lambda x: x[0].lower())

print(f'{"序":>3s}  {"谱面名称":<55s}  {"难度":>3s}  {"真实":>6s}  {"预测":>6s}  {"偏差":>7s}')
print('=' * 85)
for rank, (name, level, actual, pred) in enumerate(results, 1):
    err = pred - actual
    n = name[:55]
    print(f'{rank:>3d}  {n:<55s}  {level:>3s}  {actual:>6.2f}  {pred:>6.2f}  {err:>+7.2f}')
print('=' * 85)
print(f'共 {len(results)} 个谱面（难度 > 15.5）')
