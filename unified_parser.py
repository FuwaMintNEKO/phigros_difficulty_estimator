"""
统一谱面解析器：自动检测并转换三种谱面格式。

格式检测规则：
  1. PE格式：纯文本，不以 '{' 开头
  2. RPE标准：有 META + RPEVersion 字段，notes 存储在 'notes' 数组
  3. RPE v3 (愚人节)：单条判定线承载 ≥95% 的 notes，带移动/旋转/消失事件
  4. 官谱/标准：标准 notesAbove/notesBelow 结构
"""

import json
import math
from predict_rpe import convert_rpe_to_standard


# RPE v3 转换参数
RPEV3_VIRTUAL_LANES = 8        # 虚拟判定线数量
RPEV3_POSITION_RANGE = (-8, 8)  # 屏幕坐标范围


def detect_format(data, raw_text=None):
    """
    检测谱面格式类型。
    返回: 'pe' | 'rpe' | 'rpe_v3' | 'standard'
    """
    # PE 格式（纯文本）
    if raw_text is not None:
        text = raw_text.strip()
        if text and not text.startswith('{'):
            return 'pe'

    if not isinstance(data, dict):
        return None

    # RPE 标准格式（有 META + RPEVersion）
    if 'META' in data and 'RPEVersion' in data.get('META', {}):
        return 'rpe'

    # 检查是否为 RPE v3（愚人节单线谱）
    if _is_rpe_v3(data):
        return 'rpe_v3'

    # 标准官谱/JSON格式
    if 'judgeLineList' in data:
        return 'standard'
    if 'META' in data:
        return 'standard'

    return 'standard'


def _is_rpe_v3(data):
    """
    判断是否为 RPE v3 愚人节谱：
    1. 必须带 RPE 导出标记 numOfNotes (官方 Phigros 谱无此字段)
    2. 必须有 META.RPEVersion (RPE 程序导出必带; 官方谱永远没有)
    3. 有 judgeLineList
    4. 存在一条判定线同时满足：notes数 > 800 且有移动/旋转/消失事件

    注: 不能只用"单线音符>800"判断——现代官谱普遍由一条主线承载大部分音符
        (如 Rrharil AT 主线 1194/1300, 风屿 IN 单线 1156), 会大面积误判。
        numOfNotes 是 RPE 导出的独有字段, 官方谱 1002 张中仅 5 张含该字段且
        其主线均 <800 音符。
    但官方愚人节谱(如 Spasmodic Haocore Mix)也是单线承载大量音符(2500)且
    带 numOfNotes 字段——必须再叠加 META.RPEVersion 条件: RPE 导出的愚人节
    谱一定带该标记, 而官方谱(含愚人节谱)永远不会带, 可完全区分二者。
    """
    if 'numOfNotes' not in data:
        return False
    # 官方谱(含愚人节谱)无 META.RPEVersion; 只有 RPE 导出的谱才可能有 rpe_v3 结构
    meta = data.get('META') or {}
    if not meta.get('RPEVersion'):
        return False

    jls = data.get('judgeLineList', [])
    if not jls:
        return False

    for jl in jls:
        na = len(jl.get('notesAbove', []))
        nb = len(jl.get('notesBelow', []))
        n = na + nb
        if n > 800:
            has_events = any(k in jl for k in [
                'judgeLineMoveEvents',
                'judgeLineDisappearEvents',
                'judgeLineRotateEvents',
            ])
            if has_events:
                return True

    return False


def _convert_rpe_v3_to_standard(data):
    """
    将 RPE v3 愚人节单线谱转换为多线标准格式。
    
    策略：将所有判定线的notes按positionX均匀分配到 RPEV3_VIRTUAL_LANES 个虚拟线上。
    """
    jls = data.get('judgeLineList', [])
    if not jls:
        return data

    # 从源判定线取真实BPM
    source_bpm = 120
    for jl in jls:
        b = jl.get('bpm')
        if b and b > 0:
            source_bpm = b
            break

    # 收集所有notes（从所有线中收集，包括有事件和无事件的）
    all_notes = []
    for jl in jls:
        all_notes.extend(jl.get('notesAbove', []))
        all_notes.extend(jl.get('notesBelow', []))

    if not all_notes:
        return data

    # 确定positionX的实际范围
    positions = [n.get('positionX', 0) for n in all_notes]
    pos_min = min(positions)
    pos_max = max(positions)
    pos_range = pos_max - pos_min

    if pos_range < 0.01:
        # 所有note在同一位置，退化为单线
        result = {
            'formatVersion': 3,
            'offset': data.get('offset', 0),
            'judgeLineList': [{
                'bpm': source_bpm,
                'notesAbove': all_notes,
                'notesBelow': [],
                'speedEvents': jls[0].get('speedEvents', []) if jls else [],
            }]
        }
        if 'META' in data:
            result['META'] = data['META']
        return result

    # 将positionX范围分成N个虚拟线
    bin_width = pos_range / RPEV3_VIRTUAL_LANES

    # 初始化虚拟线
    virtual_lines = []
    for i in range(RPEV3_VIRTUAL_LANES):
        virtual_lines.append({
            'bpm': source_bpm,
            'notesAbove': [],
            'notesBelow': [],
            'speedEvents': [],
        })

    # 分配notes到虚拟线
    for note in all_notes:
        px = note.get('positionX', pos_min)
        idx = min(RPEV3_VIRTUAL_LANES - 1, max(0, int((px - pos_min) / bin_width)))
        virtual_lines[idx]['notesAbove'].append(note)

    result = {
        'formatVersion': 3,
        'offset': data.get('offset', 0),
        'judgeLineList': virtual_lines,
    }
    if 'META' in data:
        result['META'] = data['META']
    return result


def load_chart(filepath):
    """统一加载器：自动识别 PE/RPE/RPEv3/官谱，返回标准格式 dict"""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    fmt = detect_format(None, raw_text)
    if fmt == 'pe':
        return _parse_pe_format(raw_text)

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        raise ValueError(f'无法解析JSON: {filepath}')

    fmt = detect_format(data)
    if fmt == 'rpe':
        return convert_rpe_to_standard(data)
    elif fmt == 'rpe_v3':
        return _convert_rpe_v3_to_standard(data)
    elif fmt == 'standard':
        return data
    else:
        raise ValueError(f'无法识别谱面格式: {filepath}')


def load_chart_from_bytes(raw_bytes, force_format=None):
    """
    从字节数据加载，返回 (标准格式 dict, PE文本)
    force_format: 可选 'pe' | 'rpe' | 'rpe_v3' | 'standard' | None(自动)
    返回格式: (chart_data, pe_text)
    - chart_data: 标准格式dict
    - pe_text: 如果是PE格式返回原始文本，否则None
    """
    try:
        text = raw_bytes.decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError('文件编码不是 UTF-8')

    # 如果手动指定了格式，直接使用
    if force_format == 'pe':
        return _parse_pe_format(text), text
    if force_format == 'rpe':
        try:
            data = json.loads(text)
            return convert_rpe_to_standard(data), None
        except json.JSONDecodeError:
            raise ValueError('无法解析JSON格式')
    if force_format == 'rpe_v3':
        try:
            data = json.loads(text)
            return _convert_rpe_v3_to_standard(data), None
        except json.JSONDecodeError:
            raise ValueError('无法解析JSON格式')
    if force_format == 'standard':
        try:
            data = json.loads(text)
            return data, None
        except json.JSONDecodeError:
            raise ValueError('无法解析JSON格式')

    # 自动检测
    fmt = detect_format(None, text)
    if fmt == 'pe':
        return _parse_pe_format(text), text

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError('无法解析JSON格式')

    fmt = detect_format(data)
    if fmt == 'rpe':
        return convert_rpe_to_standard(data), None
    elif fmt == 'rpe_v3':
        return _convert_rpe_v3_to_standard(data), None
    elif fmt == 'standard':
        return data, None
    else:
        raise ValueError('无法识别谱面格式')


def extract_name(chart_data, raw_text=None):
    """从谱面数据提取内部名称"""
    if isinstance(chart_data, dict) and 'META' in chart_data:
        name = chart_data.get('META', {}).get('name', '')
        if name:
            return name

    if raw_text:
        lines = raw_text.strip().split('\n')
        for raw in lines[:30]:
            raw = raw.strip()
            if raw.startswith('#'):
                content = raw[1:].strip()
                if content.lower().startswith('name:') or content.lower().startswith('title:'):
                    return content.split(':', 1)[1].strip()
    return ''


def format_chart_name(filename, internal_name):
    """统一显示为 文件名 (内部名称)"""
    if internal_name and internal_name != filename:
        base = filename.rsplit('.', 1)[0] if '.' in filename else filename
        if internal_name.lower() != base.lower():
            return f'{base} ({internal_name})'
    return filename


def debug_info(data):
    """返回谱面格式的调试信息"""
    jls = data.get('judgeLineList', [])
    if not jls:
        return '无判定线'

    total_notes = sum(
        len(jl.get('notesAbove', [])) + len(jl.get('notesBelow', []))
        for jl in jls
    )
    line_counts = [
        len(jl.get('notesAbove', [])) + len(jl.get('notesBelow', []))
        for jl in jls
    ]
    non_empty = sum(1 for c in line_counts if c > 0)

    info = {
        'lines': len(jls),
        'total_notes': total_notes,
        'non_empty_lines': non_empty,
        'max_line_ratio': max(line_counts) / total_notes if total_notes > 0 else 0,
        'note_distribution': sorted(line_counts, reverse=True)[:5],
    }
    return info


# ====== PE格式解析（保持不变）======

def _parse_pe_format(text):
    """PEC 文本格式 (PhiEditer 遗留, 仅 スタートリップ / RENDA JOCEKY)

    命令一览:
      bp 时间 bpm              拍速
      n1/n2/n3/n4 线号 ...     音符 (1=Tap 2=Drag 3=Hold 4=Flick)
      cp 线号 时间 x y          判定线位置跳变 → judgeLineMoveEvents
      cm 线号 起 止 x y easing  判定线移动   → judgeLineMoveEvents
      cr 线号 起 止 角度 easing  判定线旋转   → judgeLineRotateEvents
      cf 线号 起 止 alpha       判定线透明度 → judgeLineDisappearEvents
      ca 线号 时间 alpha        判定线透明度(瞬时) → judgeLineDisappearEvents
      cv 线号 时间 速度         判定线流速   → speedEvents
      cd 线号 时间 值           (实测恒为0, 忽略)
    """
    lines = text.strip().split('\n')
    bpm = 120.0
    judge_line_count = 0
    EVENT_CMDS = ('n1', 'n2', 'n3', 'n4', 'cp', 'cv', 'cm', 'cr', 'cf', 'ca')
    bpm_list = []  # bp 行 → BPMList (PE时间单位=拍, 与标准/RPE一致)

    # 第一遍: 收集 bpm 与最大线号
    for raw in lines:
        raw = raw.strip()
        if not raw or raw.startswith('#') or raw.startswith('&'):
            continue
        parts = raw.split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd == 'bp' and len(parts) >= 3:
            bpm = float(parts[2])
            bpm_list.append({'startTime': float(parts[1]), 'bpm': bpm})
        elif cmd in EVENT_CMDS and len(parts) >= 2:
            judge_line_count = max(judge_line_count, int(parts[1]) + 1)

    if judge_line_count == 0:
        judge_line_count = 1

    judge_lines = [{'bpm': bpm, 'notesAbove': [], 'notesBelow': [], 'speedEvents': [],
                    'judgeLineMoveEvents': [], 'judgeLineRotateEvents': [],
                    'judgeLineDisappearEvents': []}
                   for _ in range(judge_line_count)]

    # PEC 格式命令 → 官方type: n1=Tap(1) n2=Hold(3) n3=Flick(4) n4=Drag(2)
    pe_type_map = {'n1': 1, 'n2': 3, 'n3': 4, 'n4': 2}
    PEC_POS_SCALE = 1024.0 / 9.0   # PEC坐标范围±1024 = 官方positionX±9
    PEC_Y_SCALE = 700.0 / 9.0      # PEC y中心300, 范围±700 → 官方positionY±9
    K = 32.0                       # 拍→ticks (与标准/RPE一致: 1拍=32ticks)
    cv_events = []                 # (line, time, value) 待合成 speedEvents

    def _move_ev(t0, t1, x, y):
        return {'startTime': t0 * K, 'endTime': t1 * K,
                'start': (x - 1024.0) / PEC_POS_SCALE, 'end': (x - 1024.0) / PEC_POS_SCALE,
                'start2': (y - 300.0) / PEC_Y_SCALE, 'end2': (y - 300.0) / PEC_Y_SCALE}

    for raw in lines:
        raw = raw.strip()
        if not raw or raw.startswith('#') or raw.startswith('&'):
            continue
        parts = raw.split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd in pe_type_map and len(parts) >= 4:
            line_idx = int(parts[1])
            if line_idx >= judge_line_count:
                continue
            start_time = float(parts[2])  # PEC时间单位=拍(beat)
            std_time = start_time * K    # 拍→ticks
            if cmd == 'n2' and len(parts) >= 5:
                # n2 长条: [n2, 线号, 起始时间, 结束时间, x(±1024), 朝向, 真假]
                end_time = float(parts[3])
                pos_x = float(parts[4])
                hold_time = max(end_time - start_time, 0) * K
                note = {'type': 3, 'time': std_time, 'positionX': pos_x / PEC_POS_SCALE,
                        'holdTime': hold_time, 'speed': 1.0}
            else:
                # n1/n3/n4 单点: [cmd, 线号, 时间, x(±1024), ...]
                pos_x = float(parts[3])
                note = {'type': pe_type_map[cmd], 'time': std_time, 'positionX': pos_x / PEC_POS_SCALE,
                        'holdTime': 0, 'speed': 1.0}
            judge_lines[line_idx]['notesAbove'].append(note)
        elif cmd == 'cm' and len(parts) >= 6:
            line_idx = int(parts[1])
            if line_idx < judge_line_count:
                t0, t1 = float(parts[2]), float(parts[3])
                judge_lines[line_idx]['judgeLineMoveEvents'].append(
                    _move_ev(t0, t1, float(parts[4]), float(parts[5])))
        elif cmd == 'cp' and len(parts) >= 5:
            line_idx = int(parts[1])
            if line_idx < judge_line_count:
                t0 = float(parts[2])
                judge_lines[line_idx]['judgeLineMoveEvents'].append(
                    _move_ev(t0, t0, float(parts[3]), float(parts[4])))
        elif cmd == 'cr' and len(parts) >= 5:
            line_idx = int(parts[1])
            if line_idx < judge_line_count:
                t0, t1, ang = float(parts[2]), float(parts[3]), float(parts[4])
                judge_lines[line_idx]['judgeLineRotateEvents'].append(
                    {'startTime': t0 * K, 'endTime': t1 * K, 'start': ang, 'end': ang})
        elif cmd == 'cf' and len(parts) >= 5:
            line_idx = int(parts[1])
            if line_idx < judge_line_count:
                t0, t1, alpha = float(parts[2]), float(parts[3]), float(parts[4])
                hide = 1 if alpha < 128 else 0
                judge_lines[line_idx]['judgeLineDisappearEvents'].append(
                    {'startTime': t0 * K, 'endTime': t1 * K, 'start': hide, 'end': hide})
        elif cmd == 'ca' and len(parts) >= 4:
            line_idx = int(parts[1])
            if line_idx < judge_line_count:
                t0, alpha = float(parts[2]), float(parts[3])
                hide = 1 if alpha < 128 else 0
                judge_lines[line_idx]['judgeLineDisappearEvents'].append(
                    {'startTime': t0 * K, 'endTime': t0 * K, 'start': hide, 'end': hide})
        elif cmd == 'cv' and len(parts) >= 4:
            line_idx = int(parts[1])
            if line_idx < judge_line_count:
                cv_events.append((line_idx, float(parts[2]) * K, float(parts[3])))

    # 合成 speedEvents: endTime 取同线下一个 cv 时间, 最后一个向后延伸 4 拍
    for line_idx in range(judge_line_count):
        evs = sorted([e for e in cv_events if e[0] == line_idx], key=lambda e: e[1])
        for i, (_, t, sp) in enumerate(evs):
            end_t = evs[i + 1][1] if i + 1 < len(evs) else t + 32.0 * 4
            judge_lines[line_idx]['speedEvents'].append(
                {'startTime': t, 'endTime': end_t, 'value': sp})

    return {'formatVersion': 3, 'BPMList': bpm_list, 'judgeLineList': judge_lines}
