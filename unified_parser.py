"""
统一谱面解析器：自动检测并转换三种谱面格式。

格式检测规则：
  1. PE格式：纯文本，不以 '{' 开头
  2. RPE标准：有 META + RPEVersion 字段，notes 存储在 'notes' 数组
  3. RPE v3 (愚人节)：单条判定线承载 ≥95% 的 notes，带移动/旋转/消失事件
  4. 官谱/标准：标准 notesAbove/notesBelow 结构
"""

import json
from predict_rpe import convert_rpe_to_standard


def detect_format(data, raw_text=None):
    """
    检测谱面格式类型。
    返回: 'pe' | 'rpe' | 'standard'
    v11.15e: 移除'rpe_v3'死代码分支 — 原 _is_rpe_v3 检测条件(numOfNotes在判定线顶层/
    音符在notes数组/事件在eventLayers)与实际RPE结构不匹配恒False, 且按positionX分虚拟线
    的策略本身错误(愚人节单线谱的多线感来自判定线时间维移动, 非空间分bin)。
    RPE单线愚人节谱统一走 'rpe' + convert_rpe_to_standard(保留单线结构+正确转换事件)。
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

    # 标准官谱/JSON格式
    if 'judgeLineList' in data:
        return 'standard'
    if 'META' in data:
        return 'standard'

    return 'standard'


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
        # v11.15e: rpe_v3合并进rpe转换(单线+标准事件), 兼容旧调用
        try:
            data = json.loads(text)
            return convert_rpe_to_standard(data), None
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
    bpm_list = []  # bp 行 → BPMList
    K = 32.0       # 拍→ticks (与标准/RPE一致: 1拍=32ticks; v11.15e: 提前定义供bp行转换用)

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
            # v11.15e: PE bp时间单位是拍, ×K=32转tick (与音符/事件时间轴统一)
            bpm_list.append({'startTime': float(parts[1]) * K, 'bpm': bpm})
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
    PEC_POS_SCALE = 1024.0 / 9.0   # 音符positionX: PEC坐标±1024 = 官方±9
    cv_events = []                 # (line, time, value) 待合成 speedEvents

    # v11.15e: PE cm命令只有终点坐标(x,y), 起点=该线上一事件的位置(初始=屏幕中心1024,700)
    # 官谱judgeLineMoveEvents的start/start2值域是[0,1]屏幕比例(0.5=中心), 不是音符的±9刻度!
    # (权威: PhiChartRender official.ts 'start: e.start - 0.5'; PE画布2048×1400 → x/2048, y/1400)
    line_pos = {}

    def _x2v(x):
        return x / 2048.0

    def _y2v(y):
        return y / 1400.0

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
                x, y = float(parts[4]), float(parts[5])
                px, py = line_pos.get(line_idx, (1024.0, 700.0))
                # v11.15e: start=前一事件位置, end=本事件终点 (原start=end=x, 位移特征恒0)
                judge_lines[line_idx]['judgeLineMoveEvents'].append(
                    {'startTime': t0 * K, 'endTime': t1 * K,
                     'start': _x2v(px), 'end': _x2v(x), 'start2': _y2v(py), 'end2': _y2v(y)})
                line_pos[line_idx] = (x, y)
        elif cmd == 'cp' and len(parts) >= 5:
            line_idx = int(parts[1])
            if line_idx < judge_line_count:
                t0 = float(parts[2])
                x, y = float(parts[3]), float(parts[4])
                judge_lines[line_idx]['judgeLineMoveEvents'].append(
                    {'startTime': t0 * K, 'endTime': t0 * K,
                     'start': _x2v(x), 'end': _x2v(x), 'start2': _y2v(y), 'end2': _y2v(y)})
                line_pos[line_idx] = (x, y)
        elif cmd == 'cr' and len(parts) >= 5:
            line_idx = int(parts[1])
            if line_idx < judge_line_count:
                t0, t1, ang = float(parts[2]), float(parts[3]), float(parts[4])
                # v11.15e: 官谱judgeLineRotateEvents单位是度(实测官谱p95=180度; 弧度只在渲染内部),
                # PE cr角度(-90~900度)直接存度, 不转弧度
                judge_lines[line_idx]['judgeLineRotateEvents'].append(
                    {'startTime': t0 * K, 'endTime': t1 * K, 'start': ang, 'end': ang})
        elif cmd == 'cf' and len(parts) >= 5:
            line_idx = int(parts[1])
            if line_idx < judge_line_count:
                t0, t1, alpha = float(parts[2]), float(parts[3]), float(parts[4])
                # v11.15e: 官谱disappear值域[0,1], 1=可见/0=消失(PhiChartRender alphaAnim语义);
                # PE alpha 0-255(255=不透明) → /255。原 hide=1 if alpha<128 else 0 语义相反(透明时写1)
                a = max(0.0, min(1.0, alpha / 255.0))
                judge_lines[line_idx]['judgeLineDisappearEvents'].append(
                    {'startTime': t0 * K, 'endTime': t1 * K, 'start': a, 'end': a})
        elif cmd == 'ca' and len(parts) >= 4:
            line_idx = int(parts[1])
            if line_idx < judge_line_count:
                t0, alpha = float(parts[2]), float(parts[3])
                a = max(0.0, min(1.0, alpha / 255.0))
                judge_lines[line_idx]['judgeLineDisappearEvents'].append(
                    {'startTime': t0 * K, 'endTime': t0 * K, 'start': a, 'end': a})
        elif cmd == 'cv' and len(parts) >= 4:
            line_idx = int(parts[1])
            if line_idx < judge_line_count:
                cv_events.append((line_idx, float(parts[2]) * K, float(parts[3])))

    # 合成 speedEvents: endTime 取同线下一个 cv 时间, 最后一个向后延伸 4 拍
    # v12: PE cv值 → 官谱倍率 = cv/10.0 (权威: phira-docs PEC文档 "速度值为float, 默认为10.000")
    # 原直接存cv原始值(xodus实测median=12)导致note_speed系列特征较官谱虚高13倍
    for line_idx in range(judge_line_count):
        evs = sorted([e for e in cv_events if e[0] == line_idx], key=lambda e: e[1])
        for i, (_, t, sp) in enumerate(evs):
            end_t = evs[i + 1][1] if i + 1 < len(evs) else t + 32.0 * 4
            judge_lines[line_idx]['speedEvents'].append(
                {'startTime': t, 'endTime': end_t, 'value': sp / 10.0})

    return {'formatVersion': 3, 'BPMList': bpm_list, 'judgeLineList': judge_lines}
