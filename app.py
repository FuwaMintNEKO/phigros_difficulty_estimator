import os, sys, json, pickle, io, zipfile, copy, numpy as np, math
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import extract_features
from unified_parser import load_chart_from_bytes
from boost_config import MANUAL_FLAT

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', '6dim_model_v12.pkl')
# v12: v11.15e特征彻查修复(30+处单位/阈值/分音/jline位移量bug)后在v11.13基线上重训

with open(MODEL_PATH, 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']
# v11.12: jline P95 修正 (训练阈值被瞬移/演出谱污染: 读谱特征几乎永不触发, Feeling Blue类被低估)
# 实际分布 P95: movement=107/rotate=18.6/disappear=15.1 (模型值 289/153/201)
_JLINE_P95_FIX = {'jline_movement_density': 107.1, 'jline_rotate_density': 18.6, 'jline_disappear_density': 15.1}
for _jk, _jv in _JLINE_P95_FIX.items():
    if _jk in P95:
        P95[_jk] = _jv
LV_ORDER = m.get('lv_order', ['EZ', 'HD', 'IN', 'AT'])
# v11.12: 权重统一用 boost_config (手工调优层; 锚点调优后pkl内旧权重作废)
MANUAL_FLAT = MANUAL_FLAT
CAPS = m.get('caps', {})  # boost excess 封顶
VERSION = f'12.0 (v11.15e彻查30+bug + PE-cv/10 + 耐力秒数 + 近似分音识别 + 双指/多指锚点) 全{ m.get("n_train", "?") }官谱'

# ===== 难点标签 (v11.7玩家研究: 官谱15+特征p75阈值) =====
_TAG_PATH = os.path.join(os.path.dirname(__file__), 'data', 'tag_thresholds.json')
try:
    with open(_TAG_PATH, encoding='utf-8') as _f:
        TAG_TH = json.load(_f)
    TAG_DIM = [('底力', 'above_avg_density_mean'), ('多押', 'weighted_mf_score_per_sec'),
               ('楼梯', 'stair_speed_avg'), ('高速', 'fast_ms_100_ratio'),
               ('爆发', 'fast_ms_050_ratio'), ('读谱', 'jline_movement_density'),
               ('变速', 'tempo_change_log_density'), ('耐力', 'above_avg_duration_sec'),
               ('高BPM', 'bpm'), ('纵连', 'jack_density'), ('叠键', 'chord_jack_3plus_pairs'),
               ('位移', 'movement_per_second'), ('锁手', 'hold_lock_weighted_per_hold')]
except Exception:
    TAG_TH = {}
    TAG_DIM = []

def compute_tags(feats):
    """按官谱15+阈值给谱面打难点标签"""
    if not TAG_TH:
        return []
    out = []
    for name, fk in TAG_DIM:
        if feats.get(fk, 0) >= TAG_TH.get(name, 1e9):
            out.append(name)
    t6 = feats.get('tracks_6plus_sec', 0) / max(feats.get('tracks_active_sec', 1), 0.01)
    if t6 >= TAG_TH.get('定轨', 1.0):
        out.append('定轨')
    return out

def _domain_warning(feats, pred):
    """v11.8: 域外/标尺提示 (玩家研究结论)"""
    warns = []
    if feats.get('drag_ratio', 0) >= 0.9 and feats.get('drag_per_sec', 0) >= 3:
        warns.append('纯drag滑动谱: 官谱无此类型, 预测仅供参考')
    if pred >= 17.0:
        warns.append('社区17+定数普遍虚高, 本预测按官谱标尺')
    if feats.get('jline_movement_density', 0) >= 300:
        warns.append('判定线表演密集: 玩家共识对其定价保守')
    return warns

# ===== kyou站玩家共识类型 (v11.8: 官谱直接引用, 自制谱分类器近似) =====
_KYOU_PATH = os.path.join(os.path.dirname(__file__), 'data', 'phira', 'kyou_tags.json')
try:
    with open(_KYOU_PATH, encoding='utf-8') as _f:
        _KYOU = json.load(_f)
except Exception:
    _KYOU = []
_KYOU_BY_NAME = {}
import re as _re
for _k in _KYOU:
    _kn = _re.sub(r'[^a-z0-9\u4e00-\u9fff]', '', _k.get('song', '').lower())
    if _kn:
        _KYOU_BY_NAME[_kn] = _k.get('tag', '').replace('?', '').strip()
_KYOU_CLF_FEATS = ['above_avg_density_mean', 'eff_avg_tps_1s', 'weighted_mf_score_per_sec', 'stair_speed_avg',
                   'thirtysecond_run_ratio', 'fast_ms_100_ratio', 'jline_movement_density', 'tempo_change_log_density',
                   'above_avg_duration_sec', 'bpm', 'jack_density', 'chord_jack_3plus_pairs', 'movement_per_second',
                   'chord_events_peak_8s', 'avg_movement', 'position_iqr', 'rhythm_entropy', 'pattern_switch_rate', 'drag_ratio']
_KYOU_CLF = None
try:
    with open(os.path.join(os.path.dirname(__file__), 'models', 'kyou_classifier.pkl'), 'rb') as _f:
        _KCLF = pickle.load(_f)
    _KYOU_CLF = _KCLF['clf']
except Exception:
    _KYOU_CLF = None

def kyou_feat_vec(feats, name='', is_custom=True):
    """v11.10: kyou标签one-hot特征 (仅官谱; 自制谱返回全0=与训练中无标签官谱分布一致)"""
    KT = ['硬抗', '综合', '定位', '读谱', '拆谱', '多指']
    if is_custom:
        return [0.0] * len(KT) + [0.0]
    kt = kyou_type_for(feats, name, is_custom)
    tag = (kt or {}).get('type', '')
    v = [0.0] * len(KT)
    if tag in KT:
        v[KT.index(tag)] = 1.0
    return v + [1.0 if tag else 0.0]  # 6类 + has_tag

def kyou_type_for(feats, name='', is_custom=True):
    """返回玩家共识类型: 官谱按kyou数据, 自制谱用分类器近似"""
    if not is_custom and name:
        kn = _re.sub(r'[^a-z0-9\u4e00-\u9fff]', '', name.lower())
        if kn in _KYOU_BY_NAME:
            return {'type': _KYOU_BY_NAME[kn], 'source': 'kyou共识'}
        for _kn, _tag in _KYOU_BY_NAME.items():
            if len(_kn) >= 4 and (_kn in kn or kn in _kn):
                return {'type': _tag, 'source': 'kyou共识'}
    if _KYOU_CLF is not None:
        x = np.array([[feats.get(k, 0) for k in _KYOU_CLF_FEATS]])
        try:
            probs = _KYOU_CLF.predict_proba(x)[0]
            cls = _KYOU_CLF.classes_[int(np.argmax(probs))]
            conf = float(np.max(probs))
            return {'type': str(cls), 'confidence': round(conf, 2), 'source': '近似'}
        except Exception:
            pass
    return None

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
MF3_SCALE_HIDENS = 0.60 # v12.2: 高密度多指谱少压→中度压 (xodus#294类多指分摊, 社区定价较物量保守)
MF3_HIDENS_TH = 9.5     # 新尺度(方案B去冗余)官谱16+段dens P50
MF3_SCALE_MID = 0.80    # 混合
EFF_SCALE_LE5 = 1.50    # 双指低密耐力型谱(dens<10): eff特征系数 (抬升)
EFF_SCALE_DF_STACK = 1.00  # 双指高密谱(dens>=10): 不抬eff (官谱双指高难dens均>=10且不抬, 保持一致; 高估Top20中18/20为双指堆料)
DF_STACK_WMF_TH = 15.0   # 双指堆料判定中心: weighted_mf_per_sec (双押宽押堆料, Breakcore 19.2 vs BonusTime 9.8)
DF_STACK_WMF_LO = 12.0   # 平滑区间下界 (wmf<12: 耐力档; 12~18: 线性过渡; >18: 堆料档)
DF_STACK_WMF_HI = 18.0   # 平滑区间上界
DF_WMF_SCALE = 0.60      # 双指堆料型: weighted_mf降权 (双押交互不算多指协调)
ML_HEAVY_TH = 100        # 多面下落型多指: multi_line_sim_events>=100 (可馅蜜协调, 非真多押)
ML_HEAVY_MF = 0.45       # 多面型: mf特征系数 (重压)
ML_HEAVY_DENS = 0.85     # 多面型: 密度特征系数
# v11.5 极端配置缩放 (AP难度视角: 官谱按AP定数, 极端配置出现即拉高)
# 双指谱: 换手/32分交互/跨线 = AP最难点 → 拉高; 多指谱: 可多指分摊/换手 → 压低(防社区多指虚高)
EXTREME_FEATS_COND = {'cross_hand_density', 'jline_relative_cross', 'thirtysecond_run_max', 'thirtysecond_run_ratio', 'lane_switch_density'}
EXTREME_SCALE_DF = 1.30   # 双指谱: 换手/32分交互是AP最难点, 温和拉高
EXTREME_SCALE_MF = 0.70   # 多指谱: 可分摊, 压低 (校准后多指仍+0.19, 强化抵抗社区虚高)
# v12: 细校准7段 (上架410首非整数定数谱MAE=0.406; 负值=抬升低估段, 正值=压低高估段)
# v12.2: PE cv单位/10 + above_avg_duration真实秒数修复后重扫; 17+段改为轻抬升(该段无校准低估-0.14)
_CALIB_TABLE = [(12, 13, -0.15), (13, 14, -0.08), (14, 15, -0.05), (15, 16, 0.04), (16, 16.5, 0.42), (16.5, 17, 0.14), (17, 99, -0.09)]  # v12.15表P: B终优化(低段解锁+社区校准层) MAE0.3887→0.3717

# ===== 社区定数校准层 (v12.12: 分段对齐社区非整数定数, 排除乱标) =====
# bins: 预测值0.5档 -> adj; 低段/17+半对齐(社区定数乱/膨胀), 16-17锚点密集区豁免(社区低标, 用户锚点已修正)
_COMM_CALIB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'community_calib.json')
try:
    with open(_COMM_CALIB_PATH, encoding='utf-8') as _f:
        _COMM_CALIB = json.load(_f).get('bins', {})
except Exception:
    _COMM_CALIB = {}


def _apply_community_calib(p):
    for _k, _v in _COMM_CALIB.items():
        _lo_s, _hi_s = _k.split('-')
        if float(_lo_s) <= p < float(_hi_s):
            return p + float(_v.get('adj', 0.0))
    return p

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
    # v11.10: 堆料降权移到 predict_one_chart 按预测段应用 (16.5+高难堆料是真难度, 不降)
    _HIGH_TAGS = {'叠键', '多押', '变速', '位移'}
    _stack_scale = 1.0
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
        if dens >= 10.0:
            eff_scale = EFF_SCALE_DF_STACK   # 高密双指: 不抬eff (官谱双指高难一致)
        else:
            eff_scale = EFF_SCALE_LE5 - (EFF_SCALE_LE5 - EFF_SCALE_DF_STACK) * _sw   # 低密耐力型: 1.5 → 1.0 平滑
        wmf_scale = 1.0 - (1.0 - DF_WMF_SCALE) * _sw                              # 1.0 → 0.6 平滑
        extreme_scale = EXTREME_SCALE_DF
    elif mf3 >= 30:
        eff_scale = 1.0
        wmf_scale = 1.0
        extreme_scale = EXTREME_SCALE_MF
    else:
        eff_scale = 1.0
        wmf_scale = 1.0
        extreme_scale = 1.0
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
            if fname in EXTREME_FEATS_COND:
                co = co * extreme_scale   # v11.5: 极端配置缩放 (AP难度视角)
            co = co * _stack_scale   # v11.8: 堆料型降权
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

def predict_from_feats(feats, level='IN', is_custom=True):
    """v12.5统一预测核心: 从(已域对齐的)特征dict → (最终定数, GB, boost, dims, contribs)
    predict_one_chart 与 CSV导出/统计脚本共用, 避免逻辑分叉(此前导出脚本复刻旧逻辑导致值不一致)"""
    x = np.array([[feats.get(n, 0) for n in FN] + _level_onehot(level)])
    xs = scaler.transform(x)
    p_gb_residual = float(gb.predict(xs)[0])

    p_boost, dims, key_contribs = compute_boost(feats, speed=1.0, is_custom=is_custom)
    p_final = p_gb_residual + p_boost
    if is_custom:
        # v11.10: 堆料降权仅中段 (14-16.5; 高难堆料不降, 修复16.5+系统性低估)
        _HIGH_TAGS = {'叠键', '多押', '变速', '位移'}
        if 14 < p_final <= 16.5 and sum(1 for t in compute_tags(feats) if t in _HIGH_TAGS) >= 2:
            p_final -= p_boost * 0.08
        # v11.1: 定轨键盘段加成 (4k/5k/6k/7k: 固定槽位密集击打, 多指分工双指无解; k数越高权重越大)
        _act = feats.get('tracks_active_sec', 0)
        if _act > 0:
            _r4 = feats.get('tracks_4plus_sec', 0) / _act
            _r5 = feats.get('tracks_5plus_sec', 0) / _act
            _r6 = feats.get('tracks_6plus_sec', 0) / _act
            _r7 = feats.get('tracks_7plus_sec', 0) / _act
            p_final += 0.15 * min(_r4, 0.8) + 0.55 * min(_r5, 0.4) + 1.0 * min(_r6, 0.15) + 1.6 * min(_r7, 0.10)
        # v11.8c: hold占比加成 (v12.5回滚保持原阶梯: hold属性本身不难)
        _hr = feats.get('hold_count', 0) / max(feats.get('total_notes', 1), 1)
        if _hr >= 0.6:
            p_final += 0.7
        elif _hr >= 0.4:
            p_final += 0.5
        elif _hr >= 0.25:
            p_final += 0.3
        # v12.6 类型规则 (官谱模型专用; 恢复版)
        _mf3t = feats.get('multi_finger_3plus_events', 0)
        _denst = feats.get('above_avg_density_mean', 0)
        if _mf3t <= 5 and _denst >= 8.0 and feats.get('odd_division_ratio', 0) >= 0.12:
            p_final -= 0.03   # 双指真底力修正(仅高奇数分音) [B终优化: -0.08->-0.03]
        elif _mf3t >= 30:
            _jmd = feats.get('jline_move_disp_per_sec', 0)
            _jrd = feats.get('jline_rotate_disp_per_sec', 0)
            if not (_jmd >= 4.5 or _jrd >= 100.0):
                p_final -= 0.02   # 单面静态多押分摊 [B终优化: -0.07->-0.02]
        _mf3r = feats.get('multi_finger_3plus_events', 0)
        _mf4r = feats.get('multi_finger_4plus_events', 0)
        _bpmr = feats.get('bpm', 0)
        _oddr = feats.get('odd_division_ratio', 0)
        _cart = feats.get('chord_alternation_rate', 0)
        _movr = feats.get('movement_per_second', 0)
        _densr = feats.get('above_avg_density_mean', 0)
        # ① 双指/轻多指物量压 (v12.9b: 加tswitch>=0.3条件 — BonusTime高速无切换16.6不应压)
        if _mf3r <= 15 and _densr >= 10.0 and _oddr < 0.12 and 170.0 <= _bpmr < 250.0 \
                and feats.get('type_switch_per_sec', 0) >= 0.3:
            p_final -= 0.40   # [B终优化: -0.48->-0.40] (v12.13: bpm>=250超高速双指不物量压 — 高仿官谱夢の降る日に误伤)
        # ①b 高速无切换双指抬 (BonusTime16.6类: 高速但蓝夹红切换少, 蓝夹红改动后boost流失)
        elif _mf3r <= 15 and _bpmr >= 220 and feats.get('type_switch_per_sec', 0) < 0.3 and _densr >= 10.0:
            p_final += 0.40   # [B终优化: +0.25->+0.40]
        # ①c 高速高切换双指抬 (v12.10: Breakcore革命前夜16.6类, 高速蓝夹红真切换; 独立规则不与①互斥)
        if _mf3r <= 5 and _bpmr >= 230 and feats.get('type_switch_per_sec', 0) >= 1.0 and _densr >= 10.0:
            p_final += 0.10   # [B终优化: +0.35->+0.10]
        # ② 切换型多指抬 (v12.9b: 加dens>=8条件 — 茉子の日常dens7.0休闲谱误伤)
        elif _mf3r >= 30 and _cart >= 2.5 and _bpmr < 170 and 8.0 <= _densr < 13.0:
            p_final += 0.40   # [B优化: +0.55->+0.40]
        # ③ 不定轨多押键盘抬 (v12.10: 多面表演型mls>=50不抬 — 八荒类表演多押虚高)
        elif _mf4r >= 50 and _movr >= 60 and feats.get('multi_line_sim_events', 0) < 50:
            p_final += 0.50
        # ④ 多指高难压
        elif _mf3r >= 80 and feats.get('note_speed_non1_ratio', 0) < 0.5 and _densr >= 12.5 \
                and (_mf4r >= 30 or _cart >= 3.8 or _densr >= 15.5)                 and not (feats.get('weighted_mf_score_per_sec', 0) >= 35.0 and _mf3r >= 200):
            p_final -= 0.48   # [B终优化: -0.30->-0.48]
        # ⑤ 高奇数分音双指压
        elif _mf3r <= 5 and _oddr >= 0.12 and _densr >= 12.0:
            p_final -= 0.32   # ⑤ 高奇数分音双指压 [B终优化]
        # ⑥ 低密长休闲谱压 (茉子の日常15.5类: dens<8长谱蓝夹红后GB抬升; 仅预测>14.5时压, 避免误伤ranked低段)
        if p_final > 14.5 and _densr < 8.0 and feats.get('duration_sec', 0) >= 90.0 and feats.get('hold_ratio', 0) < 0.85:
            p_final -= 0.32   # [B终优化: -0.35->-0.32]
        # ⑦ 表演型多指谱压 (v12.10: 线旋转/位移演出虚高; dens>=15区分Waking类低密表演; mls>=50区分FinalEndGame类单面键盘)
        _jrd7 = feats.get('jline_rotate_disp_per_sec', 0)
        _jmd7 = feats.get('jline_move_disp_per_sec', 0)
        if _mf3r >= 80 and _densr >= 15.0 and (_jrd7 >= 300.0 or _jmd7 >= 8.0):
            p_final -= 0.80 + (0.20 if _jrd7 >= 400.0 else 0.0)
        elif _mf3r >= 80 and _jrd7 < 300.0 and feats.get('movement_density_index', 0) >= 700                 and _jmd7 >= 4.5 and feats.get('multi_line_sim_events', 0) >= 50:
            p_final -= 0.40
        # ⑧ 静态暴力多指抬 (v12.10: 线静态+高多押+高密度; mf3>=200区分ギザバ怪文書18.3锚点)
        if feats.get('weighted_mf_score_per_sec', 0) >= 35.0 and _densr >= 15.0                 and _jrd7 < 60.0 and _jmd7 < 3.5 and _mf3r >= 200:
            p_final += 0.40
        # ⑨ 中等线活跃表演压 (v12.10: Xaleid#44705类, 7<=jmd<8且jrd>=80且多面, 弱于⑦a的旋转表演)
        if _mf3r >= 80 and 7.0 <= _jmd7 < 8.0 and _jrd7 >= 80.0                 and feats.get('multi_line_sim_events', 0) >= 50:
            p_final -= 0.30
        # ⑩ 静态高切换多押压 (v12.10: Chart_SP#1347类, 线静态+蓝夹红切换多+中密度多押, ④dens12.5边界差0.02)
        if _mf3r >= 80 and feats.get('type_switch_per_sec', 0) >= 1.2 and _jmd7 < 4.5                 and 10.0 <= _densr < 13.0:
            p_final -= 0.40
        # ⑪ 暴力高密度键盘抬 (v12.12: Exitium#50956/ギザバ怪文書类, bpm250+高eff高密度静态线键盘; 与⑧互斥)
        if _bpmr >= 250.0 and feats.get('eff_peak_tps_1s', 0) >= 32.0 and _densr >= 15.0                 and _jmd7 < 4.0 and _jrd7 < 60.0 and _mf3r >= 50                 and not (feats.get('weighted_mf_score_per_sec', 0) >= 35.0 and _mf3r >= 200):
            p_final += 0.40
        # v11: 预测时校准 (仅自制谱)
        for _lo, _hi, _adj in _CALIB_TABLE:
            if _lo < p_final <= _hi:
                p_final = p_final - _adj
                break
        # v12.12: 社区定数校准层 (分段对齐社区非整数定数)
        p_final = _apply_community_calib(p_final)
    return p_final, p_gb_residual, p_boost, dims, key_contribs


def predict_one_chart(chart_data, speed=1.0, level='IN', is_custom=None, chart_name=''):
    """v10.0: GB残差(含level特征) + 纯Boost叠加 = 最终定数
    is_custom: True=自制谱(应用密度域对齐), None=自动判定"""
    if is_custom is None:
        is_custom = is_custom_chart(chart_data)
    # v11.3: speed 统一为"改json"行为 — BPM缩放后全量特征(含GB)参与, 阈值不额外缩放
    if speed != 1.0:
        chart_data_scaled = apply_speed_multiplier(chart_data, speed)
        feats = extract_features(chart_data_scaled, speed=1.0)
    else:
        feats = extract_features(chart_data, speed=1.0)
    if not feats:
        return None, '特征提取失败'
    if is_custom:
        feats = apply_domain_align(feats, True, level)

    p_final, p_gb_residual, p_boost, dims, key_contribs = predict_from_feats(feats, level, is_custom)

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
        'tags': compute_tags(feats),
        'kyou_type': kyou_type_for(feats, chart_name or song_name or '', is_custom),
        'domain_warning': _domain_warning(feats, p_final),
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
        result, err = predict_one_chart(chart_data, level=level, chart_name='overlay',
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
    if level.upper() not in ('EZ', 'HD', 'IN', 'AT'):
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
                        result, err = predict_one_chart(chart_data, speed, level, chart_name=f.filename,
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
                result, err = predict_one_chart(chart_data, speed, level, chart_name=f.filename,
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
