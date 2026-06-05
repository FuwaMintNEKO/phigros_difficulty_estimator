import os, sys, json, pickle, io, zipfile, copy, numpy as np
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import extract_features
from unified_parser import load_chart_from_bytes

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', '6dim_model_v8_2.pkl')

with open(MODEL_PATH, 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']
BOOST_BINS = m.get('boost_bin_stats', {})  # 分档统计

# 从 pickle 加载 FLAT_FEATURES（与训练脚本自动同步）
FLAT_FEATURES = m.get('FLAT_FEATURES', [])
DC = m.get('dynamic_cap', {'knee': 2.5, 'power': 0.9})

import math

# v8.2: 移除峰值密度boost, target=0.28, power=0.75, thresh=0.22
RATIO_THRESHOLD = 0.22
RATIO_TARGET = 0.28
RATIO_POWER = 0.75
RATIO_STEEPNESS = 25

def adjust_boost_smooth(boost, gb):
    """Sigmoid平滑条件性调整：ratio<th不动，ratio>th逐渐施加凸性压缩。boost<2不压缩（简单谱需要boost补偿低GB）"""
    if boost < 2.0:
        return boost
    ratio = boost / gb if gb > 0 else 0
    expected = RATIO_TARGET * gb
    if expected <= 0 or boost <= 0:
        return boost
    adj = expected * ((boost / expected) ** RATIO_POWER)
    w = 1 / (1 + math.exp(-RATIO_STEEPNESS * (ratio - RATIO_THRESHOLD)))
    return (1 - w) * boost + w * adj

def _dynamic_cap(raw):
    """指数衰减cap：线性到knee，超出部分 ^power 加上去，无硬上限"""
    KNEE = DC['knee']; POWER = DC['power']
    if raw <= KNEE:
        return raw
    excess = raw - KNEE
    return KNEE + excess ** POWER


def compute_boost(feats, speed=1.0):
    """6大类别boost计算。excess指数随speed线性增加(1x=0.70, 2x=0.85)"""
    excess_exp = 0.70 + 0.15 * (speed - 1.0)
    CATEGORIES = {
        '密度': ['density_dimension', 'fast_note_density_16th', 'real_core_notes_per_second'],
        '平均位移': ['movement_per_second', 'burst_avg_movement', 'wide_jump_density', 'sim_pos_spread_max'],
        '配置': ['stair_density', 'stair_speed_avg', 'stair_complexity', 'stair_chord_ratio', 'trill_density', 'jack_density', 'chord_size_entropy', 'sim_pos_spread_mean', 'multi_finger_3plus_events', 'weighted_mf_score_per_sec', 'discrete_mf_ratio', 'chord_alternation_rate', 'position_cluster_count', 'track_deviation_score', 'position_entropy', 'position_range_used', 'pattern_switch_rate', 'direction_irregularity', 'hold_interference_index', 'drag_flick_ratio'],
        '耐力': ['stamina_ratio', 'tap_per_second', 'total_notes', 'tap_count', 'duration_sec', 'rest_ratio', 'global_jack_count', 'burst_intensity_mean', 'tap_burst_top5'],
        '读谱': ['density_transition_mean', 'density_transition_std', 'tempo_change_count', 'offbeat_ratio', 'rhythm_entropy', 'type_switch_per_sec', 'note_clutter_ratio'],
    }
    # excess指数: 1x=0.70, 速度↑→指数↑→boost响应更线性
    excess_exp = 0.70 + 0.15 * (speed - 1.0)
    # 每个类别的主要可读特征（用于显示原始值）
    CAT_RAW_KEY = {
        '密度': ('density_dimension', '(=√(TPS×峰值))'),
        '平均位移': ('movement_per_second', '格/秒'),
        '配置': ('pattern_switch_rate', '切换/秒'),
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
        x = co * (e ** excess_exp)
        if v > max(P99.get(fname, 0), bl * 0.5):
            pe = v / max(P99.get(fname, 0), bl * 0.5) - 1.0
            x += co * max(0, pe) ** excess_exp * 0.5
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


# ====== 倍速功能: 缩放 BPM ======
def apply_speed_multiplier(chart_data, speed):
    """深拷贝谱面数据，将所有BPM乘以speed倍率"""
    if speed == 1.0:
        return chart_data
    data = copy.deepcopy(chart_data)
    # 缩放每条判定线的bpm
    for jl in data.get('judgeLineList', []):
        if 'bpm' in jl:
            jl['bpm'] = jl['bpm'] * speed
    # 缩放 BPMList
    for entry in data.get('BPMList', []):
        if 'bpm' in entry:
            entry['bpm'] = entry['bpm'] * speed
    return data


# ====== 单谱预测 ======
def predict_one_chart(chart_data, speed=1.0):
    """变速预测：GB始终用1x特征，boost用变速特征"""
    # GB: 始终用 1x 特征（GB只在训练分布内有效）
    feats_1x = extract_features(chart_data, speed=1.0)
    if not feats_1x:
        return None, '特征提取失败'
    x = np.array([[feats_1x.get(n, 0) for n in FN]])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])

    # Boost: 变速特征
    if speed != 1.0:
        chart_data_scaled = apply_speed_multiplier(chart_data, speed)
        feats_boost = extract_features(chart_data_scaled, speed=speed)
    else:
        feats_boost = extract_features(chart_data, speed=1.0)

    if not feats_boost:
        return None, '特征提取失败'

    p_b, dims, key_contribs = compute_boost(feats_boost, speed=speed)
    p_b_adj = adjust_boost_smooth(p_b, p_gb)
    p_f = p_gb + p_b_adj

    # 显示用的特征值：用变速的（前端看实时数据）
    feats_display = feats_boost if speed != 1.0 else feats_1x

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
        'boost_adj': round(p_b_adj, 4),
        'boost_ratio': round(p_b / p_gb, 4) if p_gb > 0 else 0,
        'categories': dims.get('categories', {}),
        'cat_raws': dims.get('cat_raws', {}),
        'prediction': round(p_f, 4),
        'total_notes': feats_display.get('total_notes', 0),
        'duration_sec': round(feats_display.get('duration_sec', 0), 1),
        'bpm': feats_display.get('bpm', 0),
        'bpm_min': feats_display.get('bpm_min', 0),
        'bpm_max': feats_display.get('bpm_max', 0),
        'bpm_change_count': feats_display.get('bpm_change_count', 0),
        'notes_per_second': round(feats_display.get('notes_per_second', 0), 2),
        'real_notes_per_second': round(feats_display.get('real_notes_per_second', 0), 2),
        'core_notes_per_second': round(feats_display.get('core_notes_per_second', 0), 2),
        'real_core_notes_per_second': round(feats_display.get('real_core_notes_per_second', 0), 2),
        'tap_per_second': round(feats_display.get('tap_per_second', 0), 2),
        'rest_duration_sec': round(feats_display.get('duration_sec', 0) - feats_display.get('real_active_sec', 0), 1),
        'rest_ratio': round((feats_display.get('duration_sec', 0) - feats_display.get('real_active_sec', 0)) / max(feats_display.get('duration_sec', 0), 0.01), 3),
        'real_active_sec': round(feats_display.get('real_active_sec', 0), 1),
        'above_avg_density_1sec_top5': round(feats_display.get('above_avg_density_1sec_top5', 0), 1),
        'jack_count': feats_display.get('global_jack_count', 0),
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
        raw_bytes = request.get_data()
        chart_data, _ = load_chart_from_bytes(raw_bytes)
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
    # 读取用户选择的解析格式
    force_format = request.form.get('format', 'auto')
    if force_format == 'auto':
        force_format = None
    # 读取倍速参数
    try:
        speed = float(request.form.get('speed', '1.0'))
        speed = max(0.5, min(2.0, speed))  # 限制范围 0.5~2.0
    except (ValueError, TypeError):
        speed = 1.0

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
                            chart_data, raw_text = load_chart_from_bytes(raw, force_format)
                        except:
                            raw_text = raw.decode('utf-8', errors='replace')
                        if chart_data is None:
                            errors.append({'file': name, 'error': '无法解析格式'})
                            continue
                        internal_name = ''
                        if isinstance(chart_data, dict) and 'META' in chart_data:
                            internal_name = chart_data.get('META', {}).get('name', '')
                        if not internal_name and raw_text:
                            internal_name = extract_pe_name(raw_text)
                        chart_name = format_chart_name(name, internal_name)
                        result, err = predict_one_chart(chart_data, speed)
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
                    chart_data, raw_text = load_chart_from_bytes(raw_bytes, force_format)
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
                result, err = predict_one_chart(chart_data, speed)
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
