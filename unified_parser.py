import json
from predict_rpe import convert_rpe_to_standard


def load_chart(filepath):
    """统一加载器：自动识别 PE / RPE / 标准JSON，返回标准格式 dict"""
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    text = raw_text.strip()

    # 尝试 JSON 解析
    try:
        data = json.loads(text)
        if 'META' in data and 'RPEVersion' in data.get('META', {}):
            return convert_rpe_to_standard(data)
        if 'judgeLineList' in data:
            return data
        if 'META' in data:
            return data
        return data
    except json.JSONDecodeError:
        pass

    # PE 格式
    if text and not text.startswith('{'):
        return _parse_pe_format(text)

    raise ValueError(f'无法识别谱面格式: {filepath}')


def load_chart_from_bytes(raw_bytes):
    """从字节数据加载，返回标准格式 dict + PE文本(如有)"""
    text = raw_bytes.decode('utf-8')
    text_stripped = text.strip()

    try:
        data = json.loads(text_stripped)
        if 'META' in data and 'RPEVersion' in data.get('META', {}):
            return convert_rpe_to_standard(data), None
        if 'judgeLineList' in data:
            return data, None
        if 'META' in data:
            return data, None
        return data, None
    except json.JSONDecodeError:
        pass

    if text_stripped and not text_stripped.startswith('{'):
        return _parse_pe_format(text_stripped), text

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


# ====== PE格式解析 ======
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
