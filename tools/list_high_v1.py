import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import pickle
import numpy as np
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features

base_chart = os.path.join(_ROOT, 'data', 'chart')
tsv = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

diffs = load_difficulty_tsv(tsv)
charts = find_chart_files(base_chart)

with open('model_archive/v1_gb_weighted_specialist.pkl', 'rb') as f:
    model_data = pickle.load(f)

gb_model = model_data['model']
scaler = model_data['scaler']
feature_names = model_data['feature_names']
config = model_data['config']

print(f'特化双指模型: {config}')
print()

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
            X = np.array([[feats.get(n, 0) for n in feature_names]])
            X_s = scaler.transform(X)
            pred = float(gb_model.predict(X_s)[0])
            results.append((folder.replace('.0', ''), level, actual, pred))
        except:
            pass

results.sort(key=lambda x: x[0].lower())

print(f'{"序":>3s}  {"谱面名称":<55s}  {"难度":>3s}  {"真实":>6s}  {"预测":>6s}  {"偏差":>7s}')
print('=' * 85)
for rank, (name, level, actual, pred) in enumerate(results, 1):
    err = pred - actual
    n = name[:55]
    print(f'{rank:>3d}  {n:<55s}  {level:>3s}  {actual:>6.2f}  {pred:>6.2f}  {err:>+7.2f}')
print('=' * 85)
print(f'共 {len(results)} 个谱面（难度 > 15.5）')

# 分难度统计
for lv in ['EZ', 'HD', 'IN', 'AT']:
    lv_results = [(a, p) for n, l, a, p in results if l == lv]
    if lv_results:
        actuals = np.array([r[0] for r in lv_results])
        preds = np.array([r[1] for r in lv_results])
        errs = preds - actuals
        from sklearn.metrics import r2_score
        r2 = r2_score(actuals, preds)
        print(f'\n[{lv}] {len(lv_results)}个 | 平均真实={np.mean(actuals):.2f} 平均预测={np.mean(preds):.2f} | 偏差={np.mean(errs):+.3f} MAE={np.mean(np.abs(errs)):.3f} | R²={r2:.4f} | ±0.1内={np.mean(np.abs(errs)<=0.1)*100:.0f}%')

# 偏差较大的
print(f'\n偏差>0.3的谱面:')
for name, level, actual, pred in results:
    err = pred - actual
    if abs(err) > 0.3:
        print(f'  {name:<55s} {level:>3s} 真实={actual:.2f} 预测={pred:.2f} 偏差={err:+.2f}')
