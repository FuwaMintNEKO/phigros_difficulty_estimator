import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import pickle, os

base_chart = os.path.join(_ROOT, 'data', 'chart')
tsv = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

diffs = load_difficulty_tsv(tsv)
charts = find_chart_files(base_chart)

all_feats, all_labels, all_levels = [], [], []
for folder, info in charts.items():
    sid = info['song_id']
    if sid not in diffs: continue
    for level in ['EZ', 'HD', 'IN', 'AT']:
        if level not in info['levels'] or level not in diffs[sid]: continue
        try:
            data = load_chart_json(info['levels'][level])
            feats = extract_features(data)
            if feats is not None:
                all_feats.append(feats)
                all_labels.append(diffs[sid][level])
                all_levels.append(level)
        except: pass

feature_names = list(all_feats[0].keys())
X = np.array([[f.get(n, 0) for n in feature_names] for f in all_feats])
y = np.array(all_labels)
levels = np.array(all_levels)

print(f'总样本: {len(y)} (AT={sum(levels=="AT")}, IN={sum(levels=="IN")}, HD={sum(levels=="HD")}, EZ={sum(levels=="EZ")})')

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# 权重：AT 15x, IN 1.5x (最优配置)
weights = np.ones(len(y))
for i, lv in enumerate(levels):
    if lv == 'AT': weights[i] = 15.0
    elif lv == 'IN': weights[i] = 1.5

print(f'权重范围: {weights.min():.1f} ~ {weights.max():.1f}')

# GB单模型
gb = GradientBoostingRegressor(n_estimators=300, random_state=42)
gb.fit(X_s, y, sample_weight=weights)

pred = gb.predict(X_s)

at_mask = levels == 'AT'
ezhd_mask = levels != 'AT'
print(f'\nAT R² = {r2_score(y[at_mask], pred[at_mask]):.4f}')
print(f'EZ+HD+IN R² = {r2_score(y[ezhd_mask], pred[ezhd_mask]):.4f}')
print(f'Overall R² = {r2_score(y, pred):.4f}')

# 保存
model_data = {
    'model': gb,
    'scaler': scaler,
    'feature_names': feature_names,
    'config': {
        'algorithm': 'GradientBoostingRegressor',
        'n_estimators': 300,
        'weights': 'AT=15x, IN=1.5x',
        'samples': 957,
        'features': len(feature_names),
        'at_r2': float(r2_score(y[at_mask], pred[at_mask])),
        'ezhd_r2': float(r2_score(y[ezhd_mask], pred[ezhd_mask])),
        'overall_r2': float(r2_score(y, pred)),
        'description': '特化双指模型（GB单模型）。AT加权15倍、IN加权1.5倍，专精EZ+HD+IN（双指谱）评估，同时兼顾AT预测。适用场景：双指谱面难度评估。'
    }
}

save_path = 'model_archive/v1_gb_weighted_specialist.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f'\n已保存: {save_path}')
print(f'  文件大小: {os.path.getsize(save_path)/1024/1024:.1f} MB')
