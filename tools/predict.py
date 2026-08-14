import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
from feature_extractor import extract_features
from data_loader import load_chart_json


def predict_chart_difficulty(chart_data, model, level_name='unknown'):
    features = extract_features(chart_data)
    if features is None:
        print('  [错误] 无法提取特征')
        return None

    feature_values = []
    for name in model.feature_names:
        feature_values.append(features.get(name, 0))

    X = np.array([feature_values])
    prediction = model.predict(X, model_name='ensemble')

    if prediction is not None:
        return float(prediction[0])
    return None


def predict_from_file(chart_path, model, level_name='unknown'):
    try:
        chart_data = load_chart_json(chart_path)
    except Exception as e:
        print(f'  [错误] 无法读取谱面文件: {e}')
        return None

    return predict_chart_difficulty(chart_data, model, level_name)


def analyze_feature_impact(features, model):
    importance = model.get_feature_importance()
    if importance is None:
        return None

    top_features = importance[:15]
    result = []
    for name, imp in top_features:
        value = features.get(name, 0)
        result.append((name, imp, value))
    return result