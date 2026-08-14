# -*- coding: utf-8 -*-
"""未上架4.4星谱 5901张 全量预测 → 更新 unranked_4star_list.csv
(与 export_v112_predictions.py 相同预测逻辑)
"""
import os, sys, json, io, csv, time, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

with open(os.path.join(_ROOT, 'models', '6dim_model_v11_7b.pkl'), 'rb') as f:
    m = pickle.load(f)
gb, scaler = m['gb'], m['scaler']
FN, P95, P99 = m['feature_names'], m['p95_vals'], m['p99_vals']
LV_ORDER = m.get('lv_order', ['EZ','HD','IN','AT'])
CAPS = m.get('caps', {})
FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT)
_ALIGN = json.load(open(os.path.join(_ROOT, 'data', 'domain_align.json'), encoding='utf-8')).get('delta', {})
MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
DENS_FEATS = {'above_avg_density_mean', 'real_core_notes_per_second'}

def level_key(s):
    s = (s or '').upper()
    if 'AT' in s: return 'AT'
    if 'IN' in s: return 'IN'
    if 'HD' in s: return 'HD'
    return 'IN'

TAG_TH = json.load(open(os.path.join(_ROOT, 'data', 'tag_thresholds.json'), encoding='utf-8'))
TAG_DIM = [('底力', 'above_avg_density_mean'), ('多押', 'weighted_mf_score_per_sec'),
           ('楼梯', 'stair_speed_avg'), ('32分', 'thirtysecond_run_ratio'),
           ('爆发', 'fast_ms_100_ratio'), ('读谱', 'jline_movement_density'),
           ('变速', 'tempo_change_log_density'), ('耐力', 'above_avg_duration_sec'),
           ('高BPM', 'bpm'), ('纵连', 'jack_density'), ('叠键', 'chord_jack_3plus_pairs'),
           ('位移', 'movement_per_second')]
_HIGH_TAGS_SET = {'叠键', '多押', '变速', '位移'}
def _stack_scale_for(feats):
    ts = set()
    for name, fk in TAG_DIM:
        if feats.get(fk, 0) >= TAG_TH.get(name, 1e9): ts.add(name)
    if feats.get('tracks_6plus_sec', 0) / max(feats.get('tracks_active_sec', 1), 0.01) >= TAG_TH.get('定轨', 1): ts.add('定轨')
    return 0.92 if len(ts & _HIGH_TAGS_SET) >= 2 else 1.0

def predict(feats_raw, level='IN'):
    feats = dict(feats_raw)
    lv = level_key(level)
    if lv == 'IN':
        for k, d in _ALIGN.items():
            if k in feats: feats[k] = feats[k] - d
    if 'IN_AT' in LV_ORDER and lv in ('IN','AT'): lv = 'IN_AT'
    if lv not in LV_ORDER: lv = LV_ORDER[-1]
    vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(lv)] = 1.0
    x = np.array([[feats.get(n,0) for n in FN] + vec])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    mf3 = feats_raw.get('multi_finger_3plus_events', 0)
    dens = feats_raw.get('above_avg_density_mean', 0)
    ml = feats_raw.get('multi_line_sim_events', 0)
    wmf = feats_raw.get('weighted_mf_score_per_sec', 0)
    stack_scale = _stack_scale_for(feats_raw)
    if mf3 >= 30 and ml >= 100:
        mf_scale, dens_s = 0.45, 0.85
    elif mf3 >= 30:
        mf_scale, dens_s = (0.70 if dens >= 9.5 else 0.50), 1.0
    else:
        mf_scale, dens_s = (1.0 if mf3 <= 5 else 0.8), 1.0
    EXTREME = {'cross_hand_density', 'jline_relative_cross', 'thirtysecond_run_max', 'thirtysecond_run_ratio', 'lane_switch_density'}
    if mf3 <= 5:
        _sw = min(max((wmf - 12.0) / 6.0, 0.0), 1.0)
        if dens >= 10.0:
            eff_scale = 1.0
        else:
            eff_scale = 1.5 - 0.5 * _sw
        wmf_scale = 1.0 - 0.4 * _sw
        extreme_scale = 1.3
    elif mf3 >= 30:
        eff_scale, wmf_scale = 1.0, 1.0
        extreme_scale = 0.7
    else:
        eff_scale, wmf_scale = 1.0, 1.0
        extreme_scale = 1.0
    total = 0.0
    cd_ = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0)
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
        if fname in EXTREME: co2 = co * extreme_scale
        if stack_scale < 1.0: co2 = co2 * stack_scale
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
    hr = feats_raw.get('hold_count', 0) / max(feats_raw.get('total_notes', 1), 1)
    if hr >= 0.6: pred += 0.7
    elif hr >= 0.4: pred += 0.5
    elif hr >= 0.25: pred += 0.3
    for lo, hi, adj in [(14,15,0.51),(15,16,0.36),(16,17,0.16)]:
        if lo < pred <= hi: pred -= adj; break
    return pred, p_gb, total

# 元数据 (rating等)
meta = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'unranked_all.json'), encoding='utf-8'))
meta_by_id = {c['id']: c for c in meta}

JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star')
files = sorted(f for f in os.listdir(JSON_DIR) if f.endswith('.json'))
print(f'待预测: {len(files)} 张')

rows = []
ok = fail = 0
t0 = time.time()
for i, fn in enumerate(files):
    cid = int(fn[:-5])
    try:
        with open(os.path.join(JSON_DIR, fn), 'rb') as f:
            cd, raw = load_chart_from_bytes(f.read())
        feats = extract_features(cd, speed=1.0)
        if not feats:
            fail += 1; continue
        c = meta_by_id.get(cid, {})
        lv = c.get('level', 'IN')
        p, g, b = predict(feats, lv)
        rows.append([cid, c.get('name', ''), lv, c.get('difficulty'), round(c.get('rating', 0), 4), c.get('ratingCount', 0),
                     round(p, 3), round(g, 3), round(b, 3),
                     feats.get('multi_finger_3plus_events', 0), feats.get('multi_finger_4plus_events', 0),
                     round(feats.get('above_avg_density_mean', 0), 2), round(feats.get('eff_avg_tps_1s', 0), 2),
                     round(feats.get('real_core_notes_per_second', 0), 2), feats.get('total_notes', 0)])
        ok += 1
    except Exception as ex:
        fail += 1
        if fail <= 5: print(f'  失败 {cid}: {ex}')
    if (i + 1) % 500 == 0:
        el = time.time() - t0
        print(f'  {i+1}/{len(files)} (成功{ok} 失败{fail}) {el:.0f}s')
print(f'\n完成: 成功{ok} 失败{fail} 耗时{(time.time()-t0)/60:.1f}分')

# 写CSV
out_csv = os.path.join(_ROOT, 'data', 'phira', 'unranked_4star_list.csv')
with open(out_csv, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'name', 'level', 'difficulty', 'rating', 'ratingCount', 'pred', 'gb', 'boost', 'mf3', 'mf4', 'dens', 'eff_avg', 'nps', 'notes'])
    for r_ in sorted(rows, key=lambda x: -(x[4] if isinstance(x[4], (int, float)) else 0)):
        w.writerow(r_)
print(f'清单已更新: {out_csv} ({len(rows)} 行)')
print('DONE')
