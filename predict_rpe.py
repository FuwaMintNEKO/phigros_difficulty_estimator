"""
RPE格式谱面转换和预测工具
用于将RPE（另一种谱面编辑器格式）转换为Phigros标准格式并预测难度
"""

import json, os, sys, numpy as np
from feature_extractor import extract_features

RPE_TYPE_MAP = {1: 1, 2: 3, 3: 4, 4: 2}


def convert_rpe_to_standard(rpe_data):
    """将RPE格式谱面转为Phigros标准JSON格式"""
    judge_lines = rpe_data.get('judgeLineList', [])
    base_bpm = None
    for bl in rpe_data.get('BPMList', []):
        if 'bpm' in bl:
            base_bpm = bl['bpm']
            break
    if base_bpm is None:
        base_bpm = 120.0

    data = {'formatVersion': 3, 'offset': rpe_data.get('offset', 0), 'judgeLineList': []}

    # 保留 META 信息（曲名、曲师、谱师、定数等）
    if 'META' in rpe_data:
        data['META'] = rpe_data['META']

    # 保留 BPMList（变速谱的关键数据）
    bpm_list = rpe_data.get('BPMList', [])
    if bpm_list:
        data['BPMList'] = bpm_list

    for line in judge_lines:
        notes_above = []
        notes_below = []
        line['bpm'] = base_bpm
        
        for note in line.get('notes', []) if 'notes' in line else line.get('notes_display', []):
            ntype = note.get('type', 0)
            if ntype not in RPE_TYPE_MAP:
                continue
            mapped_type = RPE_TYPE_MAP[ntype]
            
            line_id = note.get('lineId', note.get('above', 0))
            is_above = line_id == 0 or line_id >= 0
            
            st = note.get('startTime', [0, 0, 1])
            if isinstance(st, (int, float)):
                start_time = float(st)
            elif isinstance(st, list):
                try:
                    measure, beat, division = st[0], st[1], st[2]
                    start_time = (float(measure) * 4.0 + float(beat) * (4.0 / float(division))) * 8.0
                except:
                    start_time = 0.0
            else:
                start_time = 0.0

            hold_time = 0.0
            if ntype == 2:
                et = note.get('endTime', None)
                if et and isinstance(et, list):
                    try:
                        em, eb, ed = et[0], et[1], et[2]
                        end_time = (float(em) * 4.0 + float(eb) * (4.0 / float(ed))) * 8.0
                        hold_time = end_time - start_time
                    except:
                        hold_time = 0.0

            speed = note.get('speed', 1.0)
            position_x = note.get('positionX', note.get('x', 0))
            position_x = float(position_x) / 100.0 if isinstance(position_x, (int, float, str)) else 0.0

            note_obj = {'type': mapped_type, 'time': start_time, 'positionX': position_x,
                        'holdTime': hold_time, 'speed': speed}

            if is_above:
                notes_above.append(note_obj)
            else:
                notes_below.append(note_obj)

        data['judgeLineList'].append({
            'bpm': base_bpm,
            'notesAbove': notes_above,
            'notesBelow': notes_below,
            'speedEvents': line.get('speedEvents', []),
        })

    return data


if __name__ == '__main__':
    from data_loader import load_chart_json

    chart_path = r'D:\迅雷下载\Phigros_Resource-master\Phigros_Resource-master\chart\4641132726938698.json'
    with open(chart_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    meta = raw_data.get('META', {})
    print(f'曲名: {meta.get("name", "?")}')
    print(f'谱师: {meta.get("charter", "?")}')
    print(f'标注难度: {meta.get("level", "?")}')
    print(f'BPM: {raw_data.get("BPMList", [{"bpm":"?"}])[0].get("bpm","?")}')
    print(f'时长: {meta.get("duration", "?")}s')

    chart_data = convert_rpe_to_standard(raw_data)

    total = 0
    total_above = 0
    total_below = 0
    total_tap = 0
    total_drag = 0
    total_hold = 0
    total_flick = 0
    total_lines = len(chart_data.get('judgeLineList', []))
    for line in chart_data.get('judgeLineList', []):
        na = line.get('notesAbove', [])
        nb = line.get('notesBelow', [])
        total_above += len(na)
        total_below += len(nb)
        for n in na + nb:
            total += 1
            t = n.get('type', 0)
            if t == 1: total_tap += 1
            elif t == 2: total_drag += 1
            elif t == 3: total_hold += 1
            elif t == 4: total_flick += 1

    print(f'  (转换映射: RPE type1→Tap, type2/4→Drag, type3→Flick)')
    print(f'\n判定线: {total_lines}条')
    print(f'总notes: {total} (Above: {total_above}, Below: {total_below})')
    print(f'Tap: {total_tap}, Drag: {total_drag}, Hold: {total_hold}, Flick: {total_flick}')

    features = extract_features(chart_data)
    if features is None:
        print('\n特征提取失败')
        exit()

    for k in ['notes_per_second', 'tap_per_second', 'duration_sec', 'duration_beats',
              'max_simultaneous', 'multi_finger_3plus_events', 'multi_finger_max_simultaneous',
              'mf_burst_count', 'wide_jump_count', 'cross_hand_event_count',
              'hold_lock_tap_events', 'hold_lock_avg_displacement', 'track_section_count',
              'micro_max_0.0625beat', 'core_micro_max_0.125beat', 'sustained_density_run_count',
              'hold_count']:
        v = features.get(k, 0)
        if v > 0:
            print(f'  {k}: {v:.4f}')

    # 使用新模型预测
    import pickle
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'gb_final_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    gb = model['gb']
    scaler = model['scaler']
    feature_names = model['feature_names']
    p95_vals = model['p95_vals']
    
    X = np.array([[features.get(n, 0) for n in feature_names]])
    xs = scaler.transform(X)
    p_gb = float(gb.predict(xs)[0])
    
    # 应用boost
    hs = features.get('hand_speed_index', 0)
    hs_p95 = p95_vals.get('hand_speed_index', 1)
    boost = 0.0
    if hs_p95 > 0 and hs > hs_p95:
        hs_ratio = hs / hs_p95
        boost += min(0.45 * float(np.log1p(hs_ratio - 1)), 0.55)
    
    p_final = p_gb + boost

    print(f'\n--- 预测结果 ---')
    print(f'  GB基础: {p_gb:.2f}')
    print(f'  外推Boost: +{boost:.3f}')
    print(f'  最终难度: {p_final:.2f}')
    
    # 特征重要性
    importance = sorted(zip(feature_names, gb.feature_importances_), key=lambda x: -x[1])[:8]
    print(f'\n--- 特征重要性 TOP 8 ---')
    for name, imp in importance:
        val = features.get(name, 0)
        print(f'  {name}: {imp:.4f} (当前值: {val:.4f})')
