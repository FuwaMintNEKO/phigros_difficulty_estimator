import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
import pickle
import json
from train_model import DifficultyModel
from feature_extractor import extract_features

chart_path = os.path.join(_ROOT, 'data', 'chart', 'Chart_SP #1347(1).json')

print(f'加载谱面: {chart_path}')
with open(chart_path, 'r', encoding='utf-8') as f:
    chart_data = json.load(f)

print(f'判定线数: {len(chart_data.get("judgeLineList", []))}')

features = extract_features(chart_data)
if features is None:
    print('特征提取失败')
    exit()

print(f'特征数: {len(features)}')
print(f'重要特征:')
for k in ['total_notes', 'tap_count', 'notes_per_second', 'tap_per_second', 'bpm', 'duration_sec', 'max_simultaneous']:
    print(f'  {k}: {features.get(k, 0):.4f}')

print('\n' + '=' * 60)
print('  当前集成模型 (v2)')
print('=' * 60)
model = DifficultyModel()
model.load('models/unified_model.pkl')
X = np.array([[features.get(n, 0) for n in model.feature_names]])
pred = model.predict(X, 'ensemble')
if pred is not None:
    pred_val = float(pred[0])
    rf_pred = model.models['random_forest'].predict(model.scaler.transform(X))[0]
    gb_pred = model.models['gradient_boosting'].predict(model.scaler.transform(X))[0]
    xgb_pred = model.models['xgboost'].predict(model.scaler.transform(X))[0] if 'xgboost' in model.models else rf_pred
    preds_list = [rf_pred, gb_pred, xgb_pred]
    ci = 1.96 * np.std(preds_list) / np.sqrt(len(preds_list))
    print(f'  预测定数: {pred_val:.2f}')
    print(f'  置信区间: ±{ci:.2f}')
    print(f'  各模型: RF={rf_pred:.2f}, GB={gb_pred:.2f}, XGB={xgb_pred:.2f}')
    print(f'  推荐难度: ', end='')
    if pred_val < 7: print(f'EZ')
    elif pred_val < 11: print(f'HD')
    elif pred_val < 14.5: print(f'IN')
    elif pred_val < 16: print(f'IN (高难)')
    else: print(f'IN/AT')

# v1 特化双指模型
print('\n' + '=' * 60)
print('  特化双指模型 (v1 GB加权)')
print('=' * 60)
with open('model_archive/v1_gb_weighted_specialist.pkl', 'rb') as f:
    v1_data = pickle.load(f)
v1_gb = v1_data['model']
v1_scaler = v1_data['scaler']
v1_names = v1_data['feature_names']
# Check features match
missing = set(v1_names) - set(features.keys())
if missing:
    print(f'  缺失 {len(missing)} 个特征（自动补0）')
X_v1 = np.array([[features.get(n, 0) for n in v1_names]])
X_v1_s = v1_scaler.transform(X_v1)
pred_v1 = float(v1_gb.predict(X_v1_s)[0])
print(f'  预测定数: {pred_v1:.2f}')
print(f'  推荐难度: ', end='')
if pred_v1 < 7: print(f'EZ')
elif pred_v1 < 11: print(f'HD')
elif pred_v1 < 14.5: print(f'IN')
elif pred_v1 < 16: print(f'IN (高难)')
else: print(f'IN/AT')

print('\n' + '=' * 60)
print('  综合结论')
print('=' * 60)
avg = (pred_val + pred_v1) / 2
print(f'  两模型平均: {avg:.2f}')
print(f'  范围: {min(pred_val, pred_v1):.2f} ~ {max(pred_val, pred_v1):.2f}')
