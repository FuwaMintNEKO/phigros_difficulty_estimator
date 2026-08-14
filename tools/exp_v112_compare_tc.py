# -*- coding: utf-8 -*-
"""test_charts 三版本并排 (统一特征+统一推理规则)"""
import os, sys, pickle, numpy as np, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
from boost_config import MANUAL_FLAT

# 加载模型
MODELS = {'v11.0': '6dim_model_v11.pkl', 'v11.1': '6dim_model_v11_1.pkl', 'v11.2': '6dim_model_v11_2.pkl'}
loaded = {}
for tag, fn in MODELS.items():
    with open(os.path.join(_ROOT, 'models', fn), 'rb') as f:
        loaded[tag] = pickle.load(f)

MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
DENS_FEATS = {'above_avg_density_mean', 'real_core_notes_per_second'}

def predict(m, feats_raw, level='IN'):
    gb, scaler = m['gb'], m['scaler']
    FN, P95, P99 = m['feature_names'], m['p95_vals'], m['p99_vals']
    LV_ORDER = m.get('lv_order', ['EZ','HD','IN','AT'])
    CAPS = m.get('caps', {})
    FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT)
    lv = level.upper()
    if 'IN_AT' in LV_ORDER and lv in ('IN','AT'): lv = 'IN_AT'
    if lv not in LV_ORDER: lv = LV_ORDER[-1]
    vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(lv)] = 1.0
    x = np.array([[feats_raw.get(n,0) for n in FN] + vec])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    mf3 = feats_raw.get('multi_finger_3plus_events', 0)
    dens = feats_raw.get('above_avg_density_mean', 0)
    ml = feats_raw.get('multi_line_sim_events', 0)
    wmf = feats_raw.get('weighted_mf_score_per_sec', 0)
    if mf3 >= 30 and ml >= 100: mf_scale, dens_s = 0.45, 0.85
    elif mf3 >= 30: mf_scale, dens_s = (0.70 if dens >= 9.5 else 0.50), 1.0
    else: mf_scale, dens_s = (1.0 if mf3 <= 5 else 0.8), 1.0
    df_stack = (mf3 <= 5 and wmf >= 15.0)
    eff_scale = 1.0 if mf3 >= 30 else (1.0 if df_stack else (1.5 if mf3 <= 5 else 1.0))
    wmf_scale = 0.6 if df_stack else 1.0
    total = 0.0
    cd_ = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats_raw.get(fname, 0)
        pv = P95.get(fname, 0)
        t = max(pv*0.55, bl*0.5)
        if v <= t: continue
        e = v/t - 1.0
        c = CAPS.get(fname, cd_)
        if c is not None and e > c: e = c
        co2 = co
        if fname in MF_FEATS: co2 = co * mf_scale
        elif fname in EFF_FEATS: co2 = co * eff_scale
        if fname in DENS_FEATS and mf3 >= 30 and ml >= 100: co2 = co * dens_s
        if fname == 'weighted_mf_score_per_sec': co2 = co * wmf_scale
        x_ = co2 * (e**0.70)
        p99 = max(P99.get(fname,0), bl*0.5)
        if v > p99:
            pe = v/p99 - 1.0
            if c is not None and pe > c: pe = c
            x_ += co2*max(0,pe)**0.70*0.5
        total += x_
    pred = p_gb + total
    act = feats_raw.get('tracks_active_sec', 0)
    if act > 0:
        r4 = feats_raw.get('tracks_4plus_sec', 0) / act
        r5 = feats_raw.get('tracks_5plus_sec', 0) / act
        r6 = feats_raw.get('tracks_6plus_sec', 0) / act
        pred += 0.15 * min(r4, 0.8) + 0.55 * min(r5, 0.4) + 1.0 * min(r6, 0.15)
    for lo, hi, adj in [(14,15,0.30),(15,16,0.18),(16,17,0.05)]:
        if lo < pred <= hi: pred -= adj; break
    return pred

def ref_from_name(fn):
    s = fn.find('(')
    if s < 0: return None
    e = fn.find(')', s)
    if e < 0: return None
    try: return float(fn[s+1:e])
    except Exception: return None

TC_DIR = os.path.join(_ROOT, 'data', 'test_charts')
rows = []
for fn in sorted(os.listdir(TC_DIR)):
    if not fn.endswith('.json'): continue
    try:
        with open(os.path.join(TC_DIR, fn), 'rb') as f:
            cd, raw = load_chart_from_bytes(f.read())
        feats = extract_features(cd, speed=1.0)
        if not feats: continue
        ps = {tag: predict(m, feats, 'IN') for tag, m in loaded.items()}
        rows.append((fn, ps))
    except Exception as ex:
        print(fn + ' ERR: ' + str(ex))

print(f'{"谱面":<44}{"v11.0":>7}{"v11.1":>7}{"v11.2":>7}{"参考":>6}')
for fn, ps in sorted(rows, key=lambda x: -x[1]['v11.2']):
    ref = ref_from_name(fn)
    ref_s = ('%.1f' % ref) if ref else '-'
    print(f'{fn[:42]:<44}{ps["v11.0"]:>7.2f}{ps["v11.1"]:>7.2f}{ps["v11.2"]:>7.2f}{ref_s:>6}')