# -*- coding: utf-8 -*-
"""v11.2 全量预测导出: 上架589 + 未上架957 → CSV
列: id, name, level, 社区定数(diff), 预测(pred), 偏差(err), mf3, dens, nps, notes, duration, gb, boost
"""
import os, sys, pickle, numpy as np, json, csv, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT

# ===== 加载模型与推理逻辑 (与 app.py v11.2 一致) =====
with open(os.path.join(_ROOT, 'models', '6dim_model_v11_2.pkl'), 'rb') as f:
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
    return 'IN'  # 未知自定义level(ST/EX/SP等)默认IN, 与app.py一致(3类模型IN/AT同向量)

def predict(feats_raw, level='IN'):
    feats = dict(feats_raw)
    lv = level_key(level)
    if lv == 'IN':  # domain align 仅 IN 段 (与 app.py 一致)
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
    if mf3 >= 30 and ml >= 100:
        mf_scale, dens_s = 0.45, 0.85
    elif mf3 >= 30:
        mf_scale, dens_s = (0.70 if dens >= 9.5 else 0.50), 1.0
    else:
        mf_scale, dens_s = (1.0 if mf3 <= 5 else 0.8), 1.0
    df_stack = (mf3 <= 5 and wmf >= 15.0)
    eff_scale = 1.0 if mf3 >= 30 else (1.0 if df_stack else (1.5 if mf3 <= 5 else 1.0))
    wmf_scale = 0.6 if df_stack else 1.0
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
    return pred, p_gb, total

def write_csv(path, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'name', 'level', 'diff', 'pred', 'err', 'gb', 'boost', 'mf3', 'mf4', 'dens', 'eff_avg', 'nps', 'notes', 'duration'])
        for r in rows:
            w.writerow(r)
    print(f'已写入: {path} ({len(rows)} 行)')

# ===== 1. 上架谱 589 (缓存特征) =====
print('=== 上架谱 ===')
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
rows = []
for r in ranked:
    p, g, b = predict(r['feats'], r['level'])
    f_ = r['feats']
    rows.append([r['id'], r['name'], r['level'], round(r['diff'], 2), round(p, 3), round(p - r['diff'], 3),
                 round(g, 3), round(b, 3),
                 f_.get('multi_finger_3plus_events', 0), f_.get('multi_finger_4plus_events', 0),
                 round(f_.get('above_avg_density_mean', 0), 2), round(f_.get('eff_avg_tps_1s', 0), 2),
                 round(f_.get('real_core_notes_per_second', 0), 2), f_.get('total_notes', 0),
                 round(f_.get('duration_sec', 0), 1)])
write_csv(os.path.join(_ROOT, 'data', 'phira', 'v112_ranked_predictions_v2.csv'), rows)

# ===== 2. 未上架谱 957 (重新提取特征) =====
print('\n=== 未上架谱 ===')
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
import csv as _csv
# 社区定数 (unranked_predictions.csv 的 diff 列)
def read_csv_cols(path):
    rows = {}
    if not os.path.exists(path): return rows
    with open(path, encoding='utf-8-sig', newline='') as f:
        rd = _csv.reader(f)
        head = next(rd)
        for c in rd:
            if len(c) < len(head): continue
            o = dict(zip(head, c))
            try: rows[int(o['id'])] = o
            except Exception: pass
    return rows
old = read_csv_cols(os.path.join(_ROOT, 'data', 'phira', 'unranked_predictions.csv'))

JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked')
rows2 = []
n_ok = n_fail = 0
for fn in sorted(os.listdir(JSON_DIR)):
    if not fn.endswith('.json'): continue
    cid = int(fn[:-5])
    try:
        with open(os.path.join(JSON_DIR, fn), 'rb') as f:
            cd, raw = load_chart_from_bytes(f.read())
        feats = extract_features(cd, speed=1.0)
        if not feats: continue
        lv = (old.get(cid) or {}).get('level', 'IN')
        p, g, b = predict(feats, lv)
        d = old.get(cid, {}).get('diff')
        d_f = float(d) if d and d not in ('', '0') else None
        rows2.append([cid, (old.get(cid) or {}).get('name', ''), lv,
                      round(d_f, 2) if d_f else '', round(p, 3),
                      round(p - d_f, 3) if d_f else '', round(g, 3), round(b, 3),
                      feats.get('multi_finger_3plus_events', 0), feats.get('multi_finger_4plus_events', 0),
                      round(feats.get('above_avg_density_mean', 0), 2), round(feats.get('eff_avg_tps_1s', 0), 2),
                      round(feats.get('real_core_notes_per_second', 0), 2), feats.get('total_notes', 0),
                      round(feats.get('duration_sec', 0), 1)])
        n_ok += 1
    except Exception as ex:
        n_fail += 1
print(f'成功: {n_ok}, 失败: {n_fail}')
write_csv(os.path.join(_ROOT, 'data', 'phira', 'v112_unranked_predictions_v2.csv'), rows2)
print('\nDONE')
