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
    1. 有 judgeLineList
    2. 存在一条判定线同时满足：notes数 > 800 且有移动/旋转/消失事件
    """
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
    lines = text.strip().split('\n')
    bpm = 120.0
    judge_line_count = 0

    for raw in lines:
        raw = raw.strip()
        if not raw or raw.startswith('#'):
            continue
        parts = raw.split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd == 'bp' and len(parts) >= 3:
            bpm = float(parts[2])
        elif cmd == 'cp' and len(parts) >= 3:
            idx = int(parts[1])
            judge_line_count = max(judge_line_count, idx + 1)

    if judge_line_count == 0:
        judge_line_count = 1

    judge_lines = [{'bpm': bpm, 'notesAbove': [], 'notesBelow': [], 'speedEvents': []}
                   for _ in range(judge_line_count)]

    pe_type_map = {'n1': 1, 'n2': 2, 'n3': 3, 'n4': 4}

    raw_hold_groups = {}

    for raw in lines:
        raw = raw.strip()
        if not raw or raw.startswith('#') or raw.startswith('&'):
            continue
        parts = raw.split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd in pe_type_map:
            ntype = pe_type_map[cmd]
            line_idx = int(parts[1])
            start_time = float(parts[2])
            pos_x = float(parts[3])
            if line_idx >= judge_line_count:
                continue
            if cmd == 'n3':
                key = (line_idx, round(pos_x, 1))
                raw_hold_groups.setdefault(key, []).append((start_time, parts))
            else:
                norm_x = pos_x / 110.0
                std_time = start_time * bpm / 1.875
                note = {'type': ntype, 'time': std_time, 'positionX': norm_x, 'holdTime': 0, 'speed': 1.0}
                if cmd == 'n2' and len(parts) >= 5:
                    note['positionY'] = float(parts[4]) / 110.0
                judge_lines[line_idx]['notesAbove'].append(note)

    for (line_idx, _), entries in raw_hold_groups.items():
        entries.sort(key=lambda x: x[0])
        used = set()
        for i in range(1, len(entries)):
            t, parts = entries[i]
            p1 = int(parts[4]) if len(parts) >= 5 else 0
            if p1 == 2:
                prev_t, prev_parts = entries[i - 1]
                prev_p1 = int(prev_parts[4]) if len(prev_parts) >= 5 else 0
                hold_duration = t - prev_t
                if prev_p1 == 1 and hold_duration >= 0.05:
                    norm_x = float(prev_parts[3]) / 110.0
                    std_time = prev_t * bpm / 1.875
                    std_hold = hold_duration * bpm / 1.875
                    note = {'type': 3, 'time': std_time, 'positionX': norm_x, 'holdTime': std_hold, 'speed': 1.0}
                    judge_lines[line_idx]['notesAbove'].append(note)
                    used.add(i - 1)
                    used.add(i)

        i = 0
        while i < len(entries):
            if i in used:
                i += 1
                continue
            t, parts = entries[i]
            pos_x = float(parts[3])
            gap = 0.50
            group = [entries[i]]
            j = i + 1
            while j < len(entries):
                if j in used or entries[j][0] - group[-1][0] > gap:
                    break
                group.append(entries[j])
                j += 1
            first_t = group[0][0]
            last_t = group[-1][0]
            hold_duration = last_t - first_t
            if hold_duration < 0.05:
                for _, gp in group:
                    norm_x = float(gp[3]) / 110.0
                    std_time = first_t * bpm / 1.875
                    if judge_line_count > 0:
                        judge_lines[line_idx]['notesAbove'].append(
                            {'type': 1, 'time': std_time, 'positionX': norm_x, 'holdTime': 0, 'speed': 1.0})
            else:
                norm_x = float(parts[3]) / 110.0
                std_time = first_t * bpm / 1.875
                std_hold = hold_duration * bpm / 1.875
                judge_lines[line_idx]['notesAbove'].append(
                    {'type': 3, 'time': std_time, 'positionX': norm_x, 'holdTime': std_hold, 'speed': 1.0})
            i = j

    return {'formatVersion': 3, 'judgeLineList': judge_lines}
