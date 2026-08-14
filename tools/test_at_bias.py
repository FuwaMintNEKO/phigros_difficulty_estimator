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

count_at = 0
at_charts = []
for folder, info in charts.items():
    sid = info['song_id']
    if sid in diffs and 'AT' in info['levels'] and 'AT' in diffs[sid]:
        at_charts.append((folder, diffs[sid]['AT']))

at_charts.sort(key=lambda x: x[1])

print("=" * 60)
print(f"AT 难度谱面共 {len(at_charts)} 个")
print("=" * 60)

preds, actuals, names = [], [], []
for folder, actual in at_charts:
    try:
        data = load_chart_json(charts[folder]['levels']['AT'])
        feats = extract_features(data)
        if feats is None: continue
        X = np.array([[feats.get(n, 0) for n in model.feature_names]])
        pred = model.predict(X, 'ensemble')
        if pred is not None:
            preds.append(float(pred[0]))
            actuals.append(actual)
            names.append(folder)
    except Exception as e:
        pass

preds = np.array(preds)
actuals = np.array(actuals)
errors = preds - actuals

print(f"\n整体统计:")
print(f"  样本数: {len(preds)}")
print(f"  平均真实值: {np.mean(actuals):.2f}")
print(f"  平均预测值: {np.mean(preds):.2f}")
print(f"  平均偏差:   {np.mean(errors):+.3f}")
print(f"  平均绝对误差: {np.mean(np.abs(errors)):.3f}")

print(f"\n按定数分段:")
bins = [(14, 15.5), (15.5, 16), (16, 16.5), (16.5, 17), (17, 17.5), (17.5, 18)]
print(f"{'定数区间':>12s}  {'数量':>4s}  {'平均真实':>8s}  {'平均预测':>8s}  {'平均偏差':>8s}")
print(f"{'-'*48}")
for lo, hi in bins:
    mask = (actuals >= lo) & (actuals < hi)
    if np.sum(mask) < 1: continue
    print(f"{lo:5.1f}-{hi:4.1f}:  {np.sum(mask):>3d}个  {np.mean(actuals[mask]):>8.2f}  {np.mean(preds[mask]):>8.2f}  {np.mean(errors[mask]):>+8.3f}")

print(f"\n所有 AT 谱面详情:")
print(f"{'谱面名称':>50s}  {'预测':>6s}  {'真实':>6s}  {'偏差':>7s}")
print(f"{'-'*75}")
idx = np.argsort(errors)
print(f"\n--- 偏低最严重的 ---")
for i in idx[:len(idx)]:
    if errors[i] >= -0.1: break
    print(f"  {names[i][:48]:48s}  {preds[i]:>6.2f}  {actuals[i]:>6.2f}  {errors[i]:>+7.2f}")

print(f"\n--- 偏高最严重的 ---")
for i in idx[-1:-(len(idx)+1):-1]:
    if errors[i] <= 0.1: break
    print(f"  {names[i][:48]:48s}  {preds[i]:>6.2f}  {actuals[i]:>6.2f}  {errors[i]:>+7.2f}")

print(f"\n--- 偏差在 ±0.1 以内的 ---")
count_good = 0
for i in range(len(preds)):
    if abs(errors[i]) <= 0.1:
        count_good += 1
        print(f"  {names[i][:48]:48s}  {preds[i]:>6.2f}  {actuals[i]:>6.2f}  {errors[i]:>+7.2f}")
print(f"\n偏差在 ±0.1 以内: {count_good}/{len(preds)} ({count_good/len(preds)*100:.1f}%)")
