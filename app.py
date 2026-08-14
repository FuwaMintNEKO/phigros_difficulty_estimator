import os, sys, json, pickle, io, zipfile, copy, numpy as np, math
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import extract_features
from unified_parser import load_chart_from_bytes
from boost_config import MANUAL_FLAT

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', '6dim_model_v11_2.pkl')
# v11.2: 方案B密度去冗余 (above_avg_density_mean改有效击打数) + 双指堆料档 + 定轨段

with open(MODEL_PATH, 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']
LV_ORDER = m.get('lv_order', ['EZ', 'HD', 'IN', 'AT'])
MANUAL_FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT)  # 优先用训练时的权重(可能含变体覆盖)
CAPS = m.get('caps', {})  # boost excess 封顶
VERSION = f'11.2 (Level-Aware GB + Boost + 密度去冗余 + 条件缩放 + 校准) 全{ m.get("n_train", "?") }官谱'

# ===== 密度域对齐 (自制谱专属, 以官谱分布为目标) =====
# 自制谱 IN(14-16.5) 密度特征系统性高于官谱同段 (domain gap, 含 drag 填充),
# 对齐 = 减去 delta[feat] (自制均值-官谱均值), 让预测回到官谱尺度。
# 数据: data/domain_align.json (70 个密度类特征), 由 tools/_tmp_gen_align.py 生成。
_ALIGN_PATH = os.path.join(os.path.dirname(__file__), 'data', 'domain_align.json')
try:
    with open(_ALIGN_PATH, encoding='utf-8') as _f:
        _ALIGN = json.load(_f)
    DOMAIN_DELTA = _ALIGN.get('delta', {})
except Exception:
    DOMAIN_DELTA = {}

def is_custom_chart(chart_data, raw_text=None):
    """判定输入是否为自制谱 (RPE/PE), 官谱 standard 格式返回 False"""
    if raw_text is not None:
        return True  # PE 文本格式
    if isinstance(chart_data, dict):
        meta = chart_data.get('META') or {}
        if meta.get('RPEVersion') is not None:
            return True  # RPE (PhiEdit) 导出
    return False

def apply_domain_align(feats, is_custom, level):
    """自制谱 IN 段: 密度特征向官谱分布对齐 (只减 delta, 不依赖社区定数)"""
    if not is_custom or not DOMAIN_DELTA:
        return feats
    if (level or 'IN').upper() != 'IN':
        return feats  # AT 段 density gap≈0/反向, 模型已低估, 不对齐
    for k, d in DOMAIN_DELTA.items():
        if k in feats:
            feats[k] = feats[k] - d
    return feats

# v11 条件boost参数: 多指谱(mf3>=30)压mf类特征, 双指谱(mf3<=5)抬eff有效单指密度
# 依据: 上架谱16+段诊断 — 多指谱被推高(社区虚高应压低), 双指谱被压低(社区偏低应抬高)
MF_FEATS_COND = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS_COND = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
DENS_FEATS_COND = {'above_avg_density_mean', 'real_core_notes_per_second'}
MF3_SCALE_GE30 = 0.50   # 低密度多指谱(堆料型): mf特征系数 (压制OOD外推虚高)
MF3_SCALE_HIDENS = 0.70 # 高密度多指谱(真材实料): 少压
MF3_HIDENS_TH = 9.5     # 新尺度(方案B去冗余)官谱16+段dens P50
MF3_SCALE_MID = 0.80    # 混合
EFF_SCALE_LE5 = 1.50    # 双指耐力型谱: eff特征系数 (抬升)
EFF_SCALE_DF_STACK = 1.00  # 双指堆料型谱: 不抬eff (t1: 高估Top20中18/20为双指堆料)
DF_STACK_WMF_TH = 15.0   # 双指堆料判定中心: weighted_mf_per_sec (双押宽押堆料, Breakcore 19.2 vs BonusTime 9.8)
DF_STACK_WMF_LO = 12.0   # 平滑区间下界 (wmf<12: 耐力档; 12~18: 线性过渡; >18: 堆料档)
DF_STACK_WMF_HI = 18.0   # 平滑区间上界
DF_WMF_SCALE = 0.60      # 双指堆料型: weighted_mf降权 (双押交互不算多指协调)
ML_HEAVY_TH = 100        # 多面下落型多指: multi_line_sim_events>=100 (可馅蜜协调, 非真多押)
ML_HEAVY_MF = 0.45       # 多面型: mf特征系数 (重压)
ML_HEAVY_DENS = 0.85     # 多面型: 密度特征系数
_CALIB_TABLE = [(14, 15, 0.30), (15, 16, 0.18), (16, 17, 0.05)]  # 预测时校准(仅自制谱, 按预测值段)

def compute_boost(feats, speed=1.0, is_custom=False):
    """v9.0: 5维纯Boost叠加，无压缩。excess指数随speed线性增加(1x=0.70, 2x=0.85)
    v11: 条件缩放(仅自制谱) — 多指谱压mf特征, 双指谱抬eff特征; 官谱保持原始权重
    v11.3: 档位判定与数值统一用feats(与改json路径一致); wmf堆料档平滑化消除跳变"""
    CATEGORIES = {
        '密度': ['real_core_notes_per_second', 'above_avg_density_mean'],
        '配置': ['stair_density', 'stair_speed_avg', 'stair_complexity', 'stair_chord_ratio', 'chord_size_entropy', 'chord_alternation_rate', 'weighted_mf_score_per_sec', 'discrete_mf_ratio', 'position_entropy', 'avg_chord_size_poly', 'position_range_used', 'trill_density', 'multi_finger_3plus_events', 'pattern_switch_rate', 'direction_irregularity', 'drag_flick_ratio'],
        '耐力': ['above_avg_duration_sec'],
        '读谱': ['tempo_change_count', 'type_switch_per_sec', 'density_transition_std', 'density_transition_mean', 'note_clutter_ratio', 'rhythm_entropy', 'hold_interference_index', 'jline_movement_density', 'jline_rotate_density', 'jline_disappear_density', 'speed_volatility', 'above_below_cross'],
    }
    excess_exp = 0.70 + 0.15 * (speed - 1.0)
    CAT_RAW_KEY = {
        '密度': ('above_avg_density_mean', '高潮段真实TPS'),
        '配置': ('stair_speed_avg', '楼梯速度/秒'),
        '耐力': ('above_avg_duration_sec', '高潮段总秒数'),
        '读谱': ('jline_movement_density', '判定线移动/秒'),
    }
    total = 0.0
    contribs = []
    cat_scores = {}
    cat_raws = {}
    cap_default = CAPS.get('_default', None)
    # v11: mf3条件缩放系数 (仅自制谱; 多指谱按密度分级; v11.2加多面型重压)
    # v11.3: 档位判定统一用feats(与改json一致); wmf堆料档平滑化(12~18线性过渡)
    mf3 = feats.get('multi_finger_3plus_events', 0)
    dens = feats.get('above_avg_density_mean', 0)
    if mf3 >= 30 and feats.get('multi_line_sim_events', 0) >= ML_HEAVY_TH:
        mf_scale = ML_HEAVY_MF       # 多面下落型(可馅蜜协调): 重压
        dens_scale_ml = ML_HEAVY_DENS
    elif mf3 >= 30:
        mf_scale = MF3_SCALE_HIDENS if dens >= MF3_HIDENS_TH else MF3_SCALE_GE30
        dens_scale_ml = 1.0
    else:
        mf_scale = 1.0 if mf3 <= 5 else MF3_SCALE_MID
    if mf3 <= 5:
        _w = feats.get('weighted_mf_score_per_sec', 0)
        _sw = min(max((_w - DF_STACK_WMF_LO) / (DF_STACK_WMF_HI - DF_STACK_WMF_LO), 0.0), 1.0)  # 0~1
        eff_scale = EFF_SCALE_LE5 - (EFF_SCALE_LE5 - EFF_SCALE_DF_STACK) * _sw   # 1.5 → 1.0 平滑
        wmf_scale = 1.0 - (1.0 - DF_WMF_SCALE) * _sw                              # 1.0 → 0.6 平滑
    else:
        eff_scale = 1.0
        wmf_scale = 1.0
    for fname, bl, co in MANUAL_FLAT:
        v = feats.get(fname, 0)
        pv = P95.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t:
            continue
        e = v / t - 1.0
        c = CAPS.get(fname, cap_default)
        if c is not None and e > c:
            e = c
        if is_custom:
            if fname in MF_FEATS_COND:
                co = co * mf_scale
            elif fname in EFF_FEATS_COND:
                co = co * eff_scale
            if fname in DENS_FEATS_COND and mf3 >= 30 and feats.get('multi_line_sim_events', 0) >= ML_HEAVY_TH:
                co = co * dens_scale_ml
            if fname == 'weighted_mf_score_per_sec':
                co = co * wmf_scale
        x = co * (e ** excess_exp)
        if v > max(P99.get(fname, 0), bl * 0.5):
            pe = v / max(P99.get(fname, 0), bl * 0.5) - 1.0
            if c is not None and pe > c:
                pe = c
            x += co * max(0, pe) ** excess_exp * 0.5
        total += x
        contribs.append((fname, round(x, 4), round(v, 2), round(t, 2), round(v/t, 3)))
    boost = total
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
def _level_onehot(level):
    """level -> one-hot (EZ/HD/IN/AT)
    若模型为 IN_AT 合并(3类)则 IN/AT 都映射到 IN_AT"""
    lv = (level or 'IN').upper()
    if 'IN_AT' in LV_ORDER and lv in ('IN', 'AT'):
        lv = 'IN_AT'
    if lv not in LV_ORDER:
        lv = 'IN_AT' if 'IN_AT' in LV_ORDER else 'IN'
    vec = [0.0] * len(LV_ORDER)
    vec[LV_ORDER.index(lv)] = 1.0
    return vec

def predict_one_chart(chart_data, speed=1.0, level='IN', is_custom=None):
    """v10.0: GB残差(含level特征) + 纯Boost叠加 = 最终定数
    is_custom: True=自制谱(应用密度域对齐), None=自动判定"""
    if is_custom is None:
        is_custom = is_custom_chart(chart_data)
    # v11.3: speed 统一为"改json"行为 — BPM缩放后全量特征(含GB)参与, 阈值不额外缩放;
    # 档位判定与数值统一用同一特征(无judge机制), wmf堆料档已平滑化
    if speed != 1.0:
        chart_data_scaled = apply_speed_multiplier(chart_data, speed)
        feats = extract_features(chart_data_scaled, speed=1.0)
    else:
        feats = extract_features(chart_data, speed=1.0)
    if not feats:
        return None, '特征提取失败'
    if is_custom:
        feats = apply_domain_align(feats, True, level)

    x = np.array([[feats.get(n, 0) for n in FN] + _level_onehot(level)])
    xs = scaler.transform(x)
    p_gb_residual = float(gb.predict(xs)[0])

    p_boost, dims, key_contribs = compute_boost(feats, speed=1.0, is_custom=is_custom)
    p_final = p_gb_residual + p_boost
    if is_custom:
        # v11.1: 定轨键盘段加成 (4k/5k/6k: 固定槽位密集击打, 多指分工双指无解; 占比归一化防长谱误伤)
        _act = feats.get('tracks_active_sec', 0)
        if _act > 0:
            _r4 = feats.get('tracks_4plus_sec', 0) / _act
            _r5 = feats.get('tracks_5plus_sec', 0) / _act
            _r6 = feats.get('tracks_6plus_sec', 0) / _act
            p_final += 0.15 * min(_r4, 0.8) + 0.55 * min(_r5, 0.4) + 1.0 * min(_r6, 0.15)
        # v11: 预测时校准 (仅自制谱: 修正社区谱口径 vs 官谱标尺的14-16段OOD高估)
        for _lo, _hi, _adj in _CALIB_TABLE:
            if _lo < p_final <= _hi:
                p_final = p_final - _adj
                break

    feats_display = feats

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
        'level_used': (level or 'IN').upper() if (level or 'IN').upper() in LV_ORDER else 'IN',
        'format': 'RPE' if is_rpe else 'Standard',
        'gb': round(p_gb_residual, 4),
        'boost': round(p_boost, 4),
        'boost_adj': round(p_boost, 4),
        'bias': 0,
        'boost_ratio': round(p_boost / p_final, 4) if p_final > 0 else 0,
        'categories': dims.get('categories', {}),
        'cat_raws': dims.get('cat_raws', {}),
        'prediction': round(p_final, 4),
        'version': VERSION,
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
        'above_avg_density_mean': round(feats_display.get('above_avg_density_mean', 0), 1),
        'above_avg_density_ratio': round(feats_display.get('above_avg_density_ratio', 0), 3),
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
    """接收单个 JSON body（原始谱面格式），返回单个预测结果（供 Android Overlay 使用）
    level 通过 query 参数 ?level=EZ|HD|IN|AT 传入"""
    try:
        raw_bytes = request.get_data()
        chart_data, raw_text = load_chart_from_bytes(raw_bytes)
        if chart_data is None:
            return jsonify({'error': '无法解析谱面格式'}), 400

        level = request.args.get('level', 'IN')
        result, err = predict_one_chart(chart_data, level=level,
                                        is_custom=is_custom_chart(chart_data, raw_text))
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
    # 读取 level 参数 (EZ/HD/IN/AT)
    level = request.form.get('level', 'IN')
    if level.upper() not in LV_ORDER:
        level = 'IN'

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
                        result, err = predict_one_chart(chart_data, speed, level,
                                                        is_custom=is_custom_chart(chart_data, raw_text))
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
                result, err = predict_one_chart(chart_data, speed, level,
                                                is_custom=is_custom_chart(chart_data, raw_text))
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
