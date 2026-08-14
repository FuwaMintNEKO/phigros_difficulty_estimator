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
model.load('models/AT_model.pkl')

preds, actuals, names, full_names = [], [], [], []
for folder, info in charts.items():
    sid = info['song_id']
    if sid in diffs and 'AT' in info['levels'] and 'AT' in diffs[sid]:
        try:
            data = load_chart_json(info['levels']['AT'])
            feats = extract_features(data)
            if feats is None: continue
            X = np.array([[feats.get(n, 0) for n in model.feature_names]])
            pred = model.predict(X, 'ensemble')
            if pred is not None:
                preds.append(float(pred[0]))
                actuals.append(diffs[sid]['AT'])
                names.append(folder.replace('.0', ''))
                full_names.append(sid)
        except:
            pass

ap = np.array(preds)
aa = np.array(actuals)
ae = ap - aa

# Sort by display name
idx = np.argsort([n.lower() for n in names])

print('=' * 100)
print('  AT 所有谱面预测结果 (按谱面名称排序)')
print('=' * 100)
print(f"{'序':>3s}  {'谱面名称 (歌曲.艺术家)':<65s}  {'预测定数':>8s}  {'真实定数':>8s}  {'偏差':>7s}")
print('-' * 100)
for rank, i in enumerate(idx, 1):
    n = names[i][:65]
    print(f"{rank:>3d}  {n:<65s}  {ap[i]:>8.2f}  {aa[i]:>8.2f}  {ae[i]:>+7.2f}")
print('-' * 100)

print(f"\n偏差 > 0.3 的谱面 (表现较差):")
for i in idx:
    if abs(ae[i]) > 0.3:
        print(f"  {names[i]:<50s}  预测={ap[i]:.2f}, 真实={aa[i]:.2f}, 偏差={ae[i]:+.2f}")

print(f"\n偏差 <= 0.1 的谱面 (表现优秀):")
good_count = 0
for i in idx:
    if abs(ae[i]) <= 0.1:
        good_count += 1
        print(f"  {names[i]:<50s}  预测={ap[i]:.2f}, 真实={aa[i]:.2f}, 偏差={ae[i]:+.2f}")
print(f"\n共 {good_count}/{len(ap)} 个偏差在 ±0.1 以内 ({good_count/len(ap)*100:.1f}%)")
