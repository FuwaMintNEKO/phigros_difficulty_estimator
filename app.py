import os, sys, json, pickle, io, zipfile, numpy as np
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import extract_features
from predict_rpe import convert_rpe_to_standard

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', '5dim_model_v5_3.pkl')

with open(MODEL_PATH, 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']

# 从 pickle 加载 FLAT_FEATURES（与训练脚本自动同步）
FLAT_FEATURES = m.get('FLAT_FEATURES', [])
DC = m.get('dynamic_cap', {'knee': 2.5, 'power': 0.9})

def _dynamic_cap(raw):
    """指数衰减cap：线性到knee，超出部分 ^power 加上去，无硬上限"""
    KNEE = DC['knee']; POWER = DC['power']
    if raw <= KNEE:
        return raw
    excess = raw - KNEE
    return KNEE + excess ** POWER


def compute_boost(feats):
    """5大类别boost计算，返回总boost、类别分数、原始值和贡献明细"""
    CATEGORIES = {
        '密度': ['core_notes_per_second', 'notes_per_second', 'peak_density_top5avg_1beat', 'density_above_zero_ratio', 'std_density_1beat'],
        '1smax密度': ['core_peak_density_1sec_top5avg', 'peak_density_1sec_top5avg', 'peak_tps_1sec_top5avg', 'micro_peak_top5_0.0625beat'],
        '平均位移': ['movement_per_second', 'burst_avg_movement', 'wide_jump_density', 'sim_pos_spread_max'],
        '耐力': ['stamina_ratio', 'tap_per_second', 'total_notes', 'tap_count', 'duration_sec', 'global_jack_count', 'burst_intensity_mean', 'tap_burst_top5'],
        '读谱': ['density_transition_mean', 'density_transition_std', 'tempo_change_count', 'offbeat_ratio', 'rhythm_entropy', 'type_switch_per_sec', 'multi_finger_3plus_events'],
    }
    # 每个类别的主要可读特征（用于显示原始值）
    CAT_RAW_KEY = {
        '密度': ('core_notes_per_second', '键/秒 (TPS)'),
        '1smax密度': ('core_peak_density_1sec_top5avg', '键/秒 (TPS)'),
        '平均位移': ('movement_per_second', '格/秒'),
        '耐力': ('tap_per_second', '键/秒'),
        '读谱': ('density_transition_mean', ''),
    }
    total = 0.0
    contribs = []
    cat_scores = {}
    cat_raws = {}
    for fname, bl, co in FLAT_FEATURES:
        v = feats.get(fname, 0)
        pv = P95.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t:
            continue
        e = v / t - 1.0
        x = co * (e ** 0.55)
        if v > max(P99.get(fname, 0), bl * 0.5):
            pe = v / max(P99.get(fname, 0), bl * 0.5) - 1.0
            x += co * max(0, pe) ** 0.55 * 0.5
        total += x
        contribs.append((fname, round(x, 4), round(v, 2), round(t, 2), round(v/t, 3)))
    boost = _dynamic_cap(total)
    # 按类别汇总
    for cat_name, feat_names in CATEGORIES.items():
        cat_sum = sum(c[1] for c in contribs if c[0] in feat_names)
        if cat_sum > 0:
            cat_scores[cat_name] = round(cat_sum, 4)
        # 代表性原始值
        raw_key, unit = CAT_RAW_KEY.get(cat_name, (None, ''))
        if raw_key:
            raw_val = feats.get(raw_key, 0)
            cat_raws[cat_name] = {'value': round(raw_val, 2), 'unit': unit}
    contribs.sort(key=lambda x: -x[1])
    return boost, {'boost': round(boost, 4), 'categories': cat_scores, 'cat_raws': cat_raws}, contribs[:15]


# ====== 统一显示名称: 文件名 (内部名称) ======
def format_chart_name(filename, internal_name):
    """统一显示为 文件名 (内部名称)，若内部名称为空则只显示文件名"""
    if internal_name and internal_name != filename:
        # 去掉文件扩展名，简约一点
        base = filename.rsplit('.', 1)[0] if '.' in filename else filename
        if internal_name.lower() != base.lower():
            return f"{base} ({internal_name})"
    return filename


# ====== PE格式: 提取谱面名称 ======
def extract_pe_name(text):
    """从PE文本开头提取 # name: / # title: 等元信息"""
    lines = text.strip().split('\n')
    for raw in lines[:30]:
        raw = raw.strip()
        if raw.startswith('#'):
            content = raw[1:].strip()
            if content.lower().startswith('name:') or content.lower().startswith('title:'):
                return content.split(':', 1)[1].strip()
    return ''


# ====== PE格式解析 (.pe / json后缀但实际是PE文本) ======
def parse_pe_format(text):
    """将PE格式文本转为Phigros标准JSON

    关键差异说明:
    - PE格式时间单位为"秒"，Phigros标准格式时间单位为"32分音符"
    - PE的n3(hold)是逐tick记录的，需要合并为单个hold带holdTime
    """
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

    judge_lines = [{'bpm': bpm, 'notesAbove': [], 'notesBelow': [], 'speedEvents': []}
                   for _ in range(judge_line_count)]

    if judge_line_count == 0:
        judge_line_count = 1
        judge_lines = [{'bpm': bpm, 'notesAbove': [], 'notesBelow': [], 'speedEvents': []}]

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
                norm_x = pos_x / 900.0
                std_time = start_time * bpm / 1.875
                note = {'type': ntype, 'time': std_time, 'positionX': norm_x, 'holdTime': 0, 'speed': 1.0}
                if cmd == 'n2' and len(parts) >= 5:
                    pos_y = float(parts[4])
                    note['positionY'] = pos_y / 900.0
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
                    norm_x = float(prev_parts[3]) / 900.0
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
            p1 = int(parts[4]) if len(parts) >= 5 else 0
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
                    pos_x = float(gp[3])
                    norm_x = pos_x / 900.0
                    std_time = first_t * bpm / 1.875
                    note = {'type': 1, 'time': std_time, 'positionX': norm_x, 'holdTime': 0, 'speed': 1.0}
                    judge_lines[line_idx]['notesAbove'].append(note)
            else:
                norm_x = float(group[0][1][3]) / 900.0
                std_time = first_t * bpm / 1.875
                std_hold = hold_duration * bpm / 1.875
                note = {'type': 3, 'time': std_time, 'positionX': norm_x, 'holdTime': std_hold, 'speed': 1.0}
                judge_lines[line_idx]['notesAbove'].append(note)
            i = j

    return {'formatVersion': 3, 'offset': 0, 'judgeLineList': judge_lines}


# ====== 解析谱面 ======
def parse_chart(data, raw_bytes=None):
    """自动识别标准/RPE/PE格式"""
    if isinstance(data, dict):
        if 'META' in data and 'RPEVersion' in data.get('META', {}):
            return convert_rpe_to_standard(data)
        return data
    if isinstance(data, str) and raw_bytes:
        text = raw_bytes.decode('utf-8')
        if text.strip() and not text.strip().startswith('{'):
            return parse_pe_format(text)
    return data


# ====== 单谱预测 ======
def predict_one_chart(chart_data):
    feats = extract_features(chart_data)
    if not feats:
        return None, '特征提取失败'

    x = np.array([[feats.get(n, 0) for n in FN]])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    p_b, dims, key_contribs = compute_boost(feats)
    p_f = p_gb + p_b

    meta = {}
    if 'META' in chart_data:
        meta = chart_data.get('META', {})
        if 'RPEVersion' in meta:
            is_rpe = True
        else:
            is_rpe = False
    else:
        is_rpe = False

    song_name = meta.get('name', '') or ''
    composer = meta.get('composer', '') or ''
    charter = meta.get('charter', '') or ''
    level_tag = meta.get('level', '') or ''

    return {
        'song_name': song_name,
        'composer': composer,
        'charter': charter,
        'level_tag': level_tag,
        'format': 'RPE' if is_rpe else 'Standard',
        'gb': round(p_gb, 4),
        'boost': round(p_b, 4),
        'categories': dims.get('categories', {}),
        'cat_raws': dims.get('cat_raws', {}),
        'prediction': round(p_f, 4),
        'total_notes': feats.get('total_notes', 0),
        'duration_sec': round(feats.get('duration_sec', 0), 1),
        'bpm': feats.get('bpm', 0),
        'notes_per_second': round(feats.get('notes_per_second', 0), 2),
        'tap_per_second': round(feats.get('tap_per_second', 0), 2),
        'jack_count': feats.get('global_jack_count', 0),
        'key_features': [
            {
                'name': fname,
                'contribution': round(c, 4),
                'value': round(v, 2),
                'threshold': round(t, 2),
                'excess': round(v/t, 3)
            }
            for fname, c, v, t, _ in key_contribs
        ],
    }, None


# ====== 路由 ======
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_one', methods=['POST'])
def predict_one():
    """接收单个 JSON body（原始谱面格式），返回单个预测结果（供 Android Overlay 使用）"""
    try:
        raw = request.get_data(as_text=True)
        if not raw:
            return jsonify({'error': '无效的 JSON'}), 400

        # 先尝试标准格式
        chart_data = None
        try:
            data = json.loads(raw)
            chart_data = parse_chart(data)
        except:
            pass

        # 再尝试 RPE 格式
        if chart_data is None:
            try:
                chart_data = parse_pe_format(raw)
            except:
                pass

        if chart_data is None:
            return jsonify({'error': '无法解析谱面格式'}), 400

        result, err = predict_one_chart(chart_data)
        if result:
            result['source_file'] = 'overlay'
            result['chart_name'] = 'overlay'
            return jsonify(result)
        else:
            return jsonify({'error': err}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    if 'files[]' not in request.files and 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400

    files = request.files.getlist('files[]') or [request.files['file']]
    results = []
    errors = []

    for f in files:
        if not f.filename:
            continue

        try:
            raw_bytes = f.read()

            if f.filename.lower().endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
                    for name in z.namelist():
                        if not name.endswith('.json') and '.' not in name.split('/')[-1]:
                            continue
                        raw = z.read(name)
                        chart_data = None
                        raw_text = None
                        try:
                            data = json.loads(raw)
                            chart_data = parse_chart(data)
                        except:
                            raw_text = raw.decode('utf-8')
                            chart_data = parse_pe_format(raw_text)
                        if chart_data is None:
                            errors.append({'file': name, 'error': '无法解析格式'})
                            continue
                        internal_name = ''
                        if isinstance(chart_data, dict) and 'META' in chart_data:
                            internal_name = chart_data.get('META', {}).get('name', '')
                        if not internal_name and raw_text:
                            internal_name = extract_pe_name(raw_text)
                        chart_name = format_chart_name(name, internal_name)
                        result, err = predict_one_chart(chart_data)
                        if result:
                            result['source_file'] = f.filename
                            result['chart_name'] = chart_name
                            results.append(result)
                        else:
                            errors.append({'file': name, 'error': err})

            else:
                chart_data = None
                raw_text = None
                try:
                    data = json.loads(raw_bytes)
                    chart_data = parse_chart(data)
                except:
                    pass
                if chart_data is None:
                    try:
                        raw_text = raw_bytes.decode('utf-8')
                        chart_data = parse_pe_format(raw_text)
                    except:
                        pass
                if chart_data is None:
                    errors.append({'file': f.filename, 'error': '无法解析谱面格式'})
                    continue
                internal_name = ''
                if isinstance(chart_data, dict) and 'META' in chart_data:
                    internal_name = chart_data.get('META', {}).get('name', '')
                if not internal_name and raw_text:
                    internal_name = extract_pe_name(raw_text)
                chart_name = format_chart_name(f.filename, internal_name)
                result, err = predict_one_chart(chart_data)
                if result:
                    result['source_file'] = f.filename
                    result['chart_name'] = chart_name
                    results.append(result)
                else:
                    errors.append({'file': f.filename, 'error': err})
        except Exception as e:
            errors.append({'file': f.filename, 'error': str(e)})

    results.sort(key=lambda x: -x['prediction'])
    return jsonify({'results': results, 'errors': errors, 'count': len(results)})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Phigros 难度预测服务器')
    parser.add_argument('--host', default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=5000, help='监听端口')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
