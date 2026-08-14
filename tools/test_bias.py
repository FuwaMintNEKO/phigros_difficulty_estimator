import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from train_model import DifficultyModel
from sklearn.metrics import r2_score
import numpy as np

base_chart = os.path.join(_ROOT, 'data', 'chart')
tsv = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

diffs = load_difficulty_tsv(tsv)
charts = find_chart_files(base_chart)

model = DifficultyModel()
model.load('models/unified_model.pkl')

print("=" * 70)
print("全统一模型测试 (EZ+HD+IN+AT)")
print("=" * 70)

results = {'preds': [], 'actuals': [], 'names': [], 'levels': []}
count = 0
total_count = 0
for folder, info in charts.items():
    sid = info['song_id']
    if sid not in diffs: continue
    for level in ['EZ', 'HD', 'IN', 'AT']:
        if level in info['levels'] and level in diffs[sid]:
            total_count += 1

for folder, info in charts.items():
    sid = info['song_id']
    if sid not in diffs: continue
    for level in ['EZ', 'HD', 'IN', 'AT']:
        if level not in info['levels'] or level not in diffs[sid]: continue
        try:
            data = load_chart_json(info['levels'][level])
            feats = extract_features(data)
            if feats is None: continue
            X = np.array([[feats.get(n, 0) for n in model.feature_names]])
            pred = model.predict(X, 'ensemble')
            if pred is not None:
                results['preds'].append(float(pred[0]))
                results['actuals'].append(diffs[sid][level])
                results['names'].append(folder.replace('.0', ''))
                results['levels'].append(level)
                count += 1
        except: pass
    if count % 300 == 0 and count > 0:
        print(f"  已处理 {count}/{total_count}...")

print(f"\n共完成 {count} 个谱面")

preds = np.array(results['preds'])
actuals = np.array(results['actuals'])
errors = preds - actuals

r2_all = r2_score(actuals, preds)

print(f"\n{'='*60}")
print(f"整体统计")
print(f"{'='*60}")
print(f"平均真实值: {np.mean(actuals):.2f}")
print(f"平均预测值: {np.mean(preds):.2f}")
print(f"平均偏差:   {np.mean(errors):+.3f}")
print(f"平均绝对误差: {np.mean(np.abs(errors)):.3f}")
print(f"R² (决定系数): {r2_all:.4f}")

print(f"\n{'='*60}")
print(f"分难度统计")
print(f"{'='*60}")
for level in ['EZ', 'HD', 'IN', 'AT']:
    mask = np.array([l == level for l in results['levels']])
    n = np.sum(mask)
    if n < 3: continue
    r2 = r2_score(actuals[mask], preds[mask])
    print(f"\n  [{level}] {n:3d}个谱面:")
    print(f"    平均真实={np.mean(actuals[mask]):.2f}  平均预测={np.mean(preds[mask]):.2f}")
    print(f"    偏差={np.mean(errors[mask]):+.3f}  MAE={np.mean(np.abs(errors[mask])):.3f}")
    print(f"    R²={r2:.4f}")
    print(f"    ±0.1以内={np.mean(np.abs(errors[mask])<=0.1)*100:.1f}%  ±0.2以内={np.mean(np.abs(errors[mask])<=0.2)*100:.1f}%")

print(f"\n{'='*60}")
print(f"按定数分段 (全难度)")
print(f"{'='*60}")
bins = [(0,3),(3,5),(5,7),(7,9),(9,11),(11,12),(12,13),(13,14),(14,15),(15,15.5),(15.5,16),(16,16.5),(16.5,17),(17,17.5),(17.5,20)]
print(f"{'定数区间':>10s} {'数量':>4s} {'真实':>6s} {'预测':>6s} {'偏差':>8s}")
for lo, hi in bins:
    mask = (actuals >= lo) & (actuals < hi)
    n = np.sum(mask)
    if n < 2: continue
    print(f"{lo:4.1f}-{hi:4.1f}: {n:4d}个 {np.mean(actuals[mask]):6.2f} {np.mean(preds[mask]):6.2f} {np.mean(errors[mask]):+8.3f}")

print(f"\n偏差最大谱面:")
idx = np.argsort(errors)
print(f"\n偏低最严重10个:")
for i in idx[:10]:
    print(f"  {results['names'][i][:40]:40s} {results['levels'][i]:3s} 预测={preds[i]:.2f}, 实际={actuals[i]:.2f}, 偏差={errors[i]:+.2f}")
print(f"\n偏高最严重10个:")
for i in idx[-10:]:
    print(f"  {results['names'][i][:40]:40s} {results['levels'][i]:3s} 预测={preds[i]:.2f}, 实际={actuals[i]:.2f}, 偏差={errors[i]:+.2f}")

print(f"\nAT 谱面详细:")
at_mask = np.array([l == 'AT' for l in results['levels']])
at_names = [results['names'][i] for i in range(len(results['names'])) if at_mask[i]]
at_preds = preds[at_mask]
at_actuals = actuals[at_mask]
at_errs = errors[at_mask]
print(f"{'序':>3s}  {'谱面名称':<55s}  {'预测':>6s}  {'真实':>6s}  {'偏差':>7s}")
print('-' * 80)
at_idx = np.argsort([n.lower() for n in at_names])
for rank, i in enumerate(at_idx, 1):
    print(f"{rank:>3d}  {at_names[i]:<55s}  {at_preds[i]:>6.2f}  {at_actuals[i]:>6.2f}  {at_errs[i]:>+7.2f}")
print(f"\nAT R² = {r2_score(at_actuals, at_preds):.4f}")
