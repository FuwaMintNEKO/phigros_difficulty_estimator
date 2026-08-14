import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os
import sys
import argparse
import numpy as np
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from train_model import DifficultyModel, print_evaluation_results

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'models')


def prepare_dataset():
    print('正在加载难度数据...')
    song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
    print(f'  加载了 {len(song_difficulties)} 首歌曲的难度信息')

    print('正在扫描谱面文件...')
    chart_files = find_chart_files(CHART_DIR)
    print(f'  找到 {len(chart_files)} 个谱面文件夹')

    by_level = {'EZ': [], 'HD': [], 'IN': [], 'AT': []}
    all_items = []

    for folder_name, info in chart_files.items():
        song_id = info['song_id']
        if song_id not in song_difficulties:
            continue

        diffs = song_difficulties[song_id]
        for level in ['EZ', 'HD', 'IN', 'AT']:
            if level in info['levels'] and level in diffs:
                item = {
                    'folder': folder_name,
                    'filepath': info['levels'][level],
                    'difficulty': diffs[level],
                    'level': level,
                }
                by_level[level].append(item)
                all_items.append(item)

    print(f'\n数据统计:')
    total_all = 0
    for level in ['EZ', 'HD', 'IN', 'AT']:
        print(f'  {level}: {len(by_level[level])} 个谱面')
        total_all += len(by_level[level])
    print(f'  EZ+HD+IN+AT (全统一模型): {total_all} 个谱面')

    return by_level, all_items


def extract_all_features(by_level, all_items):
    print('\n正在提取特征...')

    def extract_one(items, label):
        features_list = []
        labels = []
        level_tags = []
        names = []
        failed = 0
        for i, item in enumerate(items):
            try:
                chart_data = load_chart_json(item['filepath'])
                features = extract_features(chart_data)
                if features is not None:
                    features_list.append(features)
                    labels.append(item['difficulty'])
                    level_tags.append(item['level'])
                    names.append(f'{item["folder"]}_{item["level"]}')
                else:
                    failed += 1
            except:
                failed += 1
            if (i + 1) % 200 == 0:
                print(f'  [{label}] 已处理 {i+1}/{len(items)}...')
        print(f'  [{label}] 完成: 成功 {len(features_list)}, 失败 {failed}')
        return {'features': features_list, 'labels': labels, 'level_tags': level_tags, 'names': names}

    all_features = {}
    for level in ['EZ', 'HD', 'IN', 'AT']:
        all_features[level] = extract_one(by_level[level], level)

    all_features['unified'] = extract_one(all_items, 'EZ+HD+IN+AT')

    return all_features


def features_to_array(feature_list):
    if not feature_list:
        return None, None

    feature_names = sorted(feature_list[0].keys())
    X = []
    for f in feature_list:
        row = [f.get(name, 0) for name in feature_names]
        X.append(row)

    return np.array(X), feature_names


def train_all_models(all_features):
    print('\n' + '='*60)
    print('  开始训练模型')
    print('='*60)

    models = {}

    configs = [
        ('unified', 'EZ+HD+IN+AT (全统一模型)'),
    ]

    for key, display_name in configs:
        data = all_features[key]
        if len(data['features']) < 10:
            print(f'\n  [{display_name}] 数据不足, 跳过')
            continue

        X, feature_names = features_to_array(data['features'])
        if X is None:
            continue

        y = np.array(data['labels'])
        levels = data.get('level_tags', None)

        print(f'\n  [{display_name}] 训练中... (样本数: {len(y)}, 特征数: {len(feature_names)})')
        model = DifficultyModel()
        model.feature_names = feature_names
        results = model.train(X, y, display_name, levels=levels)

        print_evaluation_results(results, display_name)

        top_features = model.get_feature_importance()[:10]
        print(f'\n  最重要的10个特征:')
        for name, imp in top_features:
            print(f'    {name}: {imp:.4f}')

        models[key] = model

    return models


def cross_validate_models(all_features):
    print('\n' + '='*60)
    print('  交叉验证')
    print('='*60)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score, KFold

    for key, display_name in [('unified', 'EZ+HD+IN+AT')]:
        data = all_features[key]
        if len(data['features']) < 15:
            continue

        X, feature_names = features_to_array(data['features'])
        if X is None:
            continue

        y = np.array(data['labels'])

        rf = RandomForestRegressor(
            n_estimators=300, max_depth=20,
            min_samples_split=5, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(rf, X, y, cv=kf, scoring='neg_mean_absolute_error')
        cv_r2 = cross_val_score(rf, X, y, cv=kf, scoring='r2')

        print(f'\n  [{display_name}] 5折交叉验证:')
        print(f'    MAE: {(-cv_scores.mean()):.4f} (±{cv_scores.std():.4f})')
        print(f'    R²:  {cv_r2.mean():.4f} (±{cv_r2.std():.4f})')


def save_models(models):
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    for key in ['unified']:
        if key in models:
            save_path = os.path.join(MODEL_SAVE_PATH, 'unified_model.pkl')
            models[key].save(save_path)
            print(f'  已保存全统一模型到: {save_path}')


def load_saved_models():
    models = {}
    unified_path = os.path.join(MODEL_SAVE_PATH, 'unified_model.pkl')

    if os.path.exists(unified_path):
        model = DifficultyModel()
        model.load(unified_path)
        models['unified'] = model
        print(f'  已加载全统一模型 (EZ+HD+IN+AT)')

    return models


def predict_single_chart(chart_path, models):
    print(f'\n正在分析谱面: {chart_path}')
    try:
        chart_data = load_chart_json(chart_path)
    except Exception as e:
        print(f'  [错误] 无法读取文件: {e}')
        return

    features = extract_features(chart_data)
    if features is None:
        print('  [错误] 无法提取特征')
        return

    print(f'\n  预测结果:')
    print(f'  {"模型":>20s}  {"预测定数":>10s}  {"置信区间":>15s}')
    print(f'  {"-"*50}')

    if 'unified' in models:
        model = models['unified']
        X = np.array([[features.get(name, 0) for name in model.feature_names]])
        pred = model.predict(X, model_name='ensemble')
        if pred is not None:
            rf_pred = model.models['random_forest'].predict(model.scaler.transform(X))[0]
            xgb_pred = model.models['xgboost'].predict(model.scaler.transform(X))[0] if 'xgboost' in model.models else rf_pred
            gb_pred = model.models['gradient_boosting'].predict(model.scaler.transform(X))[0]
            preds_list = [rf_pred, gb_pred, xgb_pred]
            ci = 1.96 * np.std(preds_list) / np.sqrt(len(preds_list))
            print(f'  {"EZ/HD/IN/AT (全统一)":>20s}  {pred[0]:>8.2f}     ±{ci:.2f}')

            pred_val = float(pred[0])
            print(f'\n  综合评估:')
            if pred_val < 7:
                print(f'    推荐难度: EZ (定数约 {pred_val:.1f})')
            elif pred_val < 11:
                print(f'    推荐难度: HD (定数约 {pred_val:.1f})')
            elif pred_val < 14.5:
                print(f'    推荐难度: IN (定数约 {pred_val:.1f})')
            elif pred_val < 16:
                print(f'    推荐难度: IN (定数约 {pred_val:.1f})')
            else:
                print(f'    推荐难度: IN/AT (定数约 {pred_val:.1f})')

            importance = model.get_feature_importance()
            if importance:
                print(f'\n  主要影响特征:')
                for name, imp in importance[:8]:
                    val = features.get(name, 0)
                    print(f'    {name}: {imp:.4f} (当前值: {val:.4f})')


def main():
    parser = argparse.ArgumentParser(description='Phigros 谱面难度定数估算器')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--predict', type=str, help='预测单个谱面文件的难度', metavar='FILE')
    parser.add_argument('--retrain', action='store_true', help='重新训练（覆盖已有模型）')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        print('\n' + '='*60)
        print('  使用方式示例:')
        print('  1. 训练模型:    python main.py --train')
        print('  2. 预测谱面难度: python main.py --predict 谱面文件.json')
        return

    if args.train or args.retrain:
        by_level, all_items = prepare_dataset()
        all_features = extract_all_features(by_level, all_items)
        models = train_all_models(all_features)
        save_models(models)
        cross_validate_models(all_features)

    if args.predict:
        models = load_saved_models()
        if not models:
            print('[错误] 没有找到已训练的模型，请先运行 --train')
            return
        predict_single_chart(args.predict, models)


if __name__ == '__main__':
    main()