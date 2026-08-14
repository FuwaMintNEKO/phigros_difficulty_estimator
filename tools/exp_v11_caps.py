# -*- coding: utf-8 -*-
"""实验C: per-feature cap — 只封顶 mf/多指类特征的 excess 极端值
不动 co, 不动 GB; 对比 baseline 与各 cap 组合
"""
import os, sys, json, pickle, csv as _csv, numpy as np
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from feature_extractor import extract_features
from unified_parser import load_chart_from_bytes
from boost_config import MANUAL_FLAT

with open(os.path.join(_ROOT, 'models', '6dim_model_v10.pkl'), 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']
LV_ORDER = m.get('lv_order', ['EZ','HD','IN','AT'])
BASE_CAPS = m.get('caps', {})
FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT)

CAP_VARIANTS = {
    'baseline': {},  # cap 保持 4.0
    'v11d_mf_cap1': {'weighted_mf_score_per_sec': 1.0, 'multi_finger_3plus_events': 1.0, 'discrete_mf_ratio': 1.0},
    'v11e_mf_cap0.7': {'weighted_mf_score_per_sec': 0.7, 'multi_finger_3plus_events': 0.7, 'discrete_mf_ratio': 0.7},
    'v11f_mf_cap1_eff_up': {'weighted_mf_score_per_sec': 1.0, 'multi_finger_3plus_events': 1.0, 'discrete_mf_ratio': 1.0,
                            'eff_peak_tps_1s': 0.28, 'eff_avg_tps_1s': 0.12},
    'v11g_cap_all_high': {'weighted_mf_score_per_sec': 1.0, 'multi_finger_3plus_events': 1.0, 'discrete_mf_ratio': 1.0,
                          'stair_speed_avg': 2.0, 'chord_alternation_rate': 1.5},
}

def level_key(s):
    s = (s or '').upper()
    if 'AT' in s: return 'AT'
    if 'IN' in s: return 'IN'
    if 'HD' in s: return 'HD'
    return 'IN'

def level_onehot(lv):
    lv = level_key(lv)
    if 'IN_AT' in LV_ORDER and lv in ('IN','AT'): lv = 'IN_AT'
    if lv not in LV_ORDER: lv = LV_ORDER[-1]
    vec = [0.0]*len(LV_ORDER)
    vec[LV_ORDER.index(lv)] = 1.0
    return vec

def predict_with_caps(feats, caps, level='IN'):
    x = np.array([[feats.get(n,0) for n in FN] + level_onehot(level)])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    total = 0.0
    cd = caps.get('_default', BASE_CAPS.get('_default', 4.0))
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0)
        pv = P95.get(fname, 0)
        t = max(pv*0.55, bl*0.5)
        if v <= t: continue
        e = v/t - 1.0
        c = caps.get(fname, cd)
        if c is not None and e > c: e = c
        x_ = co * (e**0.70)
        p99 = max(P99.get(fname,0), bl*0.5)
        if v > p99:
            pe = v/p99 - 1.0
            if c is not None and pe > c: pe = c
            x_ += co*max(0,pe)**0.70*0.5
        total += x_
    return p_gb + total

# 加载
charts = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
meta_by_id = {}
for lst in charts.values():
    for c in lst:
        meta_by_id[c['id']] = c
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
pred_old = read_csv_cols(os.path.join(_ROOT, 'data', 'phira', 'predictions.csv'))

JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json')
cache = []
for fn in sorted(os.listdir(JSON_DIR)):
    if not fn.endswith('.json'): continue
    cid = int(fn[:-5])
    meta = meta_by_id.get(cid, {})
    try:
        with open(os.path.join(JSON_DIR, fn), 'rb') as f:
            chart_data, raw_text = load_chart_from_bytes(f.read())
        if chart_data is None: continue
        feats = extract_features(chart_data, speed=1.0)
        if not feats: continue
        cache.append({'id': cid, 'meta': meta, 'feats': feats})
    except Exception:
        pass
print(f'特征提取: {len(cache)}')

for vname, caps_ov in CAP_VARIANTS.items():
    caps = dict(BASE_CAPS)
    caps.update(caps_ov)
    results = []
    for item in cache:
        meta = item['meta']
        old = pred_old.get(item['id'], {})
        diff = old.get('diff')
        if not diff or float(diff) <= 10: continue
        lv = meta.get('level', 'IN')
        pred = predict_with_caps(item['feats'], caps, lv)
        results.append({'id': item['id'], 'diff': float(diff), 'pred': pred,
                        'feats': item['feats'], 'name': meta.get('name','')})
    print(f'\n===== {vname} ====')
    bins = {}
    for r in results:
        d = r['diff']
        bin_ = d < 13 and '<13' or d < 14 and '13-14' or d < 15 and '14-15' or d < 16 and '15-16' or d < 17 and '16-17' or '>=17'
        b = bins.setdefault(bin_, {'n':0,'b':0,'mae':0})
        b['n'] += 1; b['b'] += r['pred']-d; b['mae'] += abs(r['pred']-d)
    for k in sorted(bins, key=lambda x: float(x.replace('<','0').replace('-','.').replace('>=','99'))):
        b = bins[k]
        print(f'  {k}: n={b["n"]} bias={b["b"]/b["n"]:+.3f} MAE={b["mae"]/b["n"]:.3f}')
    hi = [r for r in results if r['diff'] >= 16]
    groups = {}
    for r in hi:
        mf3 = r['feats'].get('multi_finger_3plus_events', 0)
        g = '多指(mf3>=30)' if mf3 >= 30 else ('双指(mf3<=5)' if mf3 <= 5 else '混合')
        gr = groups.setdefault(g, {'n':0,'b':0,'mae':0})
        gr['n'] += 1; gr['b'] += r['pred']-r['diff']; gr['mae'] += abs(r['pred']-r['diff'])
    print('  16+ 分组:')
    for g, gr in groups.items():
        print(f'    {g}: n={gr["n"]} bias={gr["b"]/gr["n"]:+.3f} MAE={gr["mae"]/gr["n"]:.3f}')
print('\nDONE')
