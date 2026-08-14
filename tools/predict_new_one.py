import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import json
import numpy as np
from feature_extractor import extract_features
from train_model import DifficultyModel

chart_path = os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json')

model = DifficultyModel()
model.load('models/unified_model.pkl')

with open(chart_path, 'r', encoding='utf-8') as f:
    chart_data = json.load(f)

lines = chart_data.get('judgeLineList', [])
print(f'判定线数: {len(lines)}')

features = extract_features(chart_data)
if features is None:
    print('特征提取失败')
else:
    tap_notes = [n for n in sum((l.get('notesAbove',[]) + l.get('notesBelow',[]) for l in lines), []) if n.get('type')==1]
    drag_notes = [n for n in sum((l.get('notesAbove',[]) + l.get('notesBelow',[]) for l in lines), []) if n.get('type')==2]
    hold_notes = [n for n in sum((l.get('notesAbove',[]) + l.get('notesBelow',[]) for l in lines), []) if n.get('type')==3]
    flick_notes = [n for n in sum((l.get('notesAbove',[]) + l.get('notesBelow',[]) for l in lines), []) if n.get('type')==4]

    print(f'Tap: {len(tap_notes)}, Drag: {len(drag_notes)}, Hold: {len(hold_notes)}, Flick: {len(flick_notes)}')
    print(f'总notes: {len(tap_notes)+len(drag_notes)+len(hold_notes)+len(flick_notes)}')

    for k in ['notes_per_second', 'tap_per_second', 'bpm', 'duration_sec', 'duration_beats',
              'max_simultaneous', 'multi_finger_3plus_events', 'multi_finger_max_simultaneous',
              'mf_burst_count', 'mf_with_hold_count', 'wide_jump_count', 'cross_hand_event_count',
              'hold_lock_tap_events', 'hold_lock_avg_displacement', 'track_section_count',
              'micro_max_0.0625beat', 'core_micro_max_0.125beat', 'sustained_density_run_count']:
        v = features.get(k, 0)
        if v > 0:
            print(f'  {k}: {v:.4f}')

    X = np.array([[features.get(n, 0) for n in model.feature_names]])

    print(f'\n--- 预测结果 ---')
    rf_pred = model.models['random_forest'].predict(model.scaler.transform(X))[0]
    gb_pred = model.models['gradient_boosting'].predict(model.scaler.transform(X))[0]
    xgb_pred = model.models['xgboost'].predict(model.scaler.transform(X))[0] if 'xgboost' in model.models else rf_pred
    preds_list = [rf_pred, gb_pred, xgb_pred]
    ci = 1.96 * np.std(preds_list) / np.sqrt(len(preds_list))

    weight_sum = 1.0 + 3.0 + 1.0
    ensemble_raw = (rf_pred * 1.0 + gb_pred * 3.0 + xgb_pred * 1.0) / weight_sum
    ensemble_cal = ensemble_raw + model.calibration_slope * ensemble_raw + model.calibration_intercept

    print(f'  Random Forest: {rf_pred:.2f}')
    print(f'  Gradient Boosting: {gb_pred:.2f}')
    print(f'  XGBoost: {xgb_pred:.2f}')
    print(f'  集成模型 (加权): {ensemble_cal:.2f}  ±{ci:.2f}')

    print(f'\n  --- 特征重要性 TOP 8 ---')
    importance = model.get_feature_importance()
    for name, imp in importance[:8]:
        val = features.get(name, 0)
        print(f'    {name}: {imp:.4f} (当前值: {val:.4f})')

    pred_val = float(ensemble_cal)
    print(f'\n  综合评估:')
    if pred_val < 7:
        print(f'    推荐难度: EZ (定数约 {pred_val:.1f})')
    elif pred_val < 11:
        print(f'    推荐难度: HD (定数约 {pred_val:.1f})')
    elif pred_val < 14.5:
        print(f'    推荐难度: IN (定数约 {pred_val:.1f})')
    else:
        print(f'    推荐难度: IN/AT (定数约 {pred_val:.1f})')
