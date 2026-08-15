"""
RPE格式谱面转换和预测工具
用于将RPE（另一种谱面编辑器格式）转换为Phigros标准格式并预测难度
"""

import json, os, sys, numpy as np
from feature_extractor import extract_features

RPE_TYPE_MAP = {1: 1, 2: 3, 3: 4, 4: 2}  # v11.8c修复: RPE type2=Hold(带endTime, 需长按)→标准3; type3=Flick(瞬时少位置)→标准4; type4=Drag(瞬时多位置)→标准2
# 铁证1: 3rd Avenue type=2 128个音符 endTime-startTime 全>0(均值3.36拍) = 长条(Hold)
# 铁证2: 用户实测 Feeling Blue (47264) 游戏内全部为Hold长按 — 旧映射{2:2}把它解析成Drag导致完全失明(预测10.9 vs 实际≈16)
# 与PE映射同构: pe_type_map {'n1':1,'n2':3,'n3':4,'n4':2}


import bisect


def _rpe_time_to_ticks(t):
    """RPE 时间 [m,b,d] → 官谱tick = (m + b/d)*32
    与音符公式 (m*4 + b*(4/d))*8 数值恒等(m是拍不是小节, 1拍=32tick); 数字直接当tick。"""
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, (list, tuple)) and len(t) >= 3:
        try:
            m, b, d = float(t[0]), float(t[1]), float(t[2])
            if d == 0:
                d = 1.0
            return (m + b / d) * 32.0
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _convert_rpe_speed_events(line):
    """RPE speedEvents → 官谱标准 {startTime(tick), endTime(tick), value(倍率)}
    权威转换(PhiChartRender rephiedit.ts:141-142): value = rpe_value / (0.6 / (120/900)) = rpe_value / 4.5
    已是官谱格式(带value字段)的事件只转时间; start≠end的渐变拆成两段阶跃近似。"""
    out = []
    raw_events = list(line.get('speedEvents', []))
    for layer in line.get('eventLayers', []) or []:
        if layer:
            raw_events.extend(layer.get('speedEvents', []) or [])
    for ev in raw_events:
        t0 = _rpe_time_to_ticks(ev.get('startTime'))
        t1 = _rpe_time_to_ticks(ev.get('endTime'))
        if t1 <= t0:
            t1 = t0 + 32.0 * 4
        if 'value' in ev:
            out.append({'startTime': t0, 'endTime': t1, 'value': float(ev.get('value', 1.0))})
            continue
        start_v = float(ev.get('start', 4.5))
        end_v = float(ev.get('end', start_v))
        out.append({'startTime': t0, 'endTime': t1, 'value': start_v / 4.5})
        if abs(end_v - start_v) > 1e-6:
            mid = (t0 + t1) / 2.0
            out[-1]['endTime'] = mid
            out.append({'startTime': mid, 'endTime': t1, 'value': end_v / 4.5})
    return out


def _convert_rpe_event_layers(line):
    """RPE eventLayers → 官谱 judgeLineMoveEvents/judgeLineRotateEvents/judgeLineDisappearEvents
    权威转换(PhiChartRender rephiedit.ts:144-165):
      moveX: px/1350 → 官谱start值域[0,1] (中心0.5 = 675px/1350 + 0.5; 与official.ts的start-0.5互逆)
      moveY: px/900  → 官谱start2
      rotate: 度 → 官谱度 (官谱JSON格式本身用度; 弧度只存在于渲染内部表示)
      alpha: /255 clip[-1,1] → 官谱disappear (值域[0,1], 1=可见/0=消失)
    moveX与moveY是独立事件列表, 按合并时间轴线性插值保证x/y同步。"""
    move_x, move_y = [], []
    rotate_evs, alpha_evs = [], []
    for layer in line.get('eventLayers', []) or []:
        if not layer:
            continue
        for ev in layer.get('moveXEvents', []) or []:
            move_x.append((_rpe_time_to_ticks(ev.get('startTime')), _rpe_time_to_ticks(ev.get('endTime')),
                           float(ev.get('start', 0)), float(ev.get('end', 0))))
        for ev in layer.get('moveYEvents', []) or []:
            move_y.append((_rpe_time_to_ticks(ev.get('startTime')), _rpe_time_to_ticks(ev.get('endTime')),
                           float(ev.get('start', 0)), float(ev.get('end', 0))))
        for ev in layer.get('rotateEvents', []) or []:
            rotate_evs.append({'startTime': _rpe_time_to_ticks(ev.get('startTime')),
                               'endTime': _rpe_time_to_ticks(ev.get('endTime')),
                               'start': float(ev.get('start', 0)),  # 度→度(官谱格式即度)
                               'end': float(ev.get('end', 0))})
        for ev in layer.get('alphaEvents', []) or []:
            a0 = max(-1.0, min(1.0, float(ev.get('start', 255)) / 255.0))
            a1 = max(-1.0, min(1.0, float(ev.get('end', 255)) / 255.0))
            alpha_evs.append({'startTime': _rpe_time_to_ticks(ev.get('startTime')),
                              'endTime': _rpe_time_to_ticks(ev.get('endTime')),
                              'start': a0, 'end': a1})

    move_evs = []
    if move_x or move_y:
        move_x_sorted = sorted(move_x, key=lambda e: e[0])
        move_y_sorted = sorted(move_y, key=lambda e: e[0])

        def make_sampler(events_sorted):
            starts = [e[0] for e in events_sorted]
            def sample(t):
                idx = bisect.bisect_right(starts, t) - 1
                if idx < 0:
                    return events_sorted[0][2] if events_sorted else 0.0
                if idx >= len(events_sorted):
                    return events_sorted[-1][3]
                t0, t1, v0, v1 = events_sorted[idx]
                if t >= t1:
                    return v1
                span = t1 - t0
                return v0 if span <= 1e-9 else v0 + (v1 - v0) * (t - t0) / span
            return sample

        sx = make_sampler(move_x_sorted)
        sy = make_sampler(move_y_sorted)
        time_points = sorted(set(e[0] for e in move_x + move_y) | set(e[1] for e in move_x + move_y))
        for i in range(1, len(time_points)):
            ta, tb = time_points[i - 1], time_points[i]
            if tb <= ta:
                continue
            move_evs.append({'startTime': ta, 'endTime': tb,
                             'start': sx(ta) / 1350.0 + 0.5, 'end': sx(tb) / 1350.0 + 0.5,
                             'start2': sy(ta) / 900.0 + 0.5, 'end2': sy(tb) / 900.0 + 0.5})

    return move_evs, rotate_evs, alpha_evs


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

    # BPMList → 官谱标准: startTime转tick (原样保留[m,b,d]会被下游按tick消费时错位)
    bpm_list = []
    for bl in rpe_data.get('BPMList', []) or []:
        b = bl.get('bpm')
        if b is None:
            continue
        bpm_list.append({'startTime': _rpe_time_to_ticks(bl.get('startTime', [0, 0, 1])), 'bpm': b})
    if bpm_list:
        data['BPMList'] = bpm_list

    for line in judge_lines:
        notes_above = []
        notes_below = []
        line['bpm'] = base_bpm
        
        for note in line.get('notes', []) if 'notes' in line else line.get('notes_display', []):
            # RPE 假音符(isFake=1)是装饰/演出用, 实际不可玩, 必须过滤, 否则密度/物量虚高
            if int(note.get('isFake', 0) or 0) == 1:
                continue
            ntype = note.get('type', 0)
            if ntype not in RPE_TYPE_MAP:
                continue
            mapped_type = RPE_TYPE_MAP[ntype]
            
            # RPE above 语义: 1=正面(判定线前方), 0/2=背面 → 只有 above==1 归 notesAbove
            # 注意新版RPE的 lineId key 存在但值为 None, 不能作为判定依据
            is_above = note.get('above', 1) == 1
            
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
            # RPE positionX 是像素坐标(视口±675), 官方为±9 → 除以 675/9=75
            position_x = float(position_x) / 75.0 if isinstance(position_x, (int, float, str)) else 0.0

            note_obj = {'type': mapped_type, 'time': start_time, 'positionX': position_x,
                        'holdTime': hold_time, 'speed': speed}

            if is_above:
                notes_above.append(note_obj)
            else:
                notes_below.append(note_obj)

        # v11.15e: eventLayers转换为官谱标准事件(move/rotate/disappear), 不再原样保留
        move_evs, rotate_evs, alpha_evs = _convert_rpe_event_layers(line)
        new_line = {
            'bpm': base_bpm,
            'notesAbove': notes_above,
            'notesBelow': notes_below,
            'speedEvents': _convert_rpe_speed_events(line),
            'judgeLineMoveEvents': move_evs,
            'judgeLineRotateEvents': rotate_evs,
            'judgeLineDisappearEvents': alpha_evs,
        }
        if line.get('extended'):
            new_line['extended'] = line['extended']
        data['judgeLineList'].append(new_line)

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
