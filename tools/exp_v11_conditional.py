# -*- coding: utf-8 -*-
"""实验D: 推理层条件boost — mf3条件衰减 + eff抬升 (不重训, 验证净效果)
- 多指谱(mf3>=30): mf特征co x0.3
- 双指谱(mf3<=5): mf特征co x1.0 (不动), eff co x1.3
- 混合: mf x0.7
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
CAPS = m.get('caps', {})
FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT)

# 条件boost参数
MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
MF_SCALE = {30: 0.40, 5: 1.0}  # mf3>=30 时 mf特征 x0.4
EFF_SCALE = {30: 1.0, 5: 1.50}  # 双指谱 eff x1.5

_ALIGN = {}
try:
    with open(os.path.join(_ROOT, 'data', 'domain_align.json'), encoding='utf-8') as _f:
        _ALIGN = json.load(_f).get('delta', {})
except Exception:
    pass

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

def compute_boost(feats, mf3):
    if level_key('IN') == 'IN':
        pass
    mf_scale = MF_SCALE.get(30, 1.0) if mf3 >= 30 else (MF_SCALE.get(5, 1.0) if mf3 <= 5 else 0.8)
    eff_scale = EFF_SCALE.get(30, 1.0) if mf3 >= 30 else (EFF_SCALE.get(5, 1.0) if mf3 <= 5 else 1.0)
    total = 0.0
    cd = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0)
        pv = P95.get(fname, 0)
        t = max(pv*0.55, bl*0.5)
        if v <= t: continue
        e = v/t - 1.0
        c = CAPS.get(fname, cd)
        if c is not None and e > c: e = c
        co2 = co
        if fname in MF_FEATS: co2 = co * mf_scale
        elif fname in EFF_FEATS: co2 = co * eff_scale
        x_ = co2 * (e**0.70)
        p99 = max(P99.get(fname,0), bl*0.5)
        if v > p99:
            pe = v/p99 - 1.0
            if c is not None and pe > c: pe = c
            x_ += co2*max(0,pe)**0.70*0.5
        total += x_
    return total

def predict(feats, level):
    feats2 = dict(feats)
    if level_key(level) == 'IN':
        for k, d in _ALIGN.items():
            if k in feats2: feats2[k] = feats2[k] - d
    x = np.array([[feats2.get(n,0) for n in FN] + level_onehot(level)])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    mf3 = feats.get('multi_finger_3plus_events', 0)
    b = compute_boost(feats2, mf3)
    return p_gb + b

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

results = []
for item in cache:
    meta = item['meta']
    old = pred_old.get(item['id'], {})
    diff = old.get('diff')
    if not diff or float(diff) <= 10: continue
    lv = meta.get('level', 'IN')
    pred = predict(item['feats'], lv)
    results.append({'id': item['id'], 'diff': float(diff), 'pred': pred,
                    'feats': item['feats'], 'name': meta.get('name','')})

print(f'\n===== 实验D: 条件boost (mf3衰减+eff抬升) ====')
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

# ===== 外推段排序 =====
ext = [r for r in results if r['diff'] >= 17.7]
print('\n=== 外推段 (社区>=17.7) ===')
for r in sorted(ext, key=lambda x: -x['diff']):
    mf3 = r['feats'].get('multi_finger_3plus_events', 0)
    print(f'  {r["name"][:26]}: 社区 {r["diff"]:.1f} | 预测 {r["pred"]:.2f} | 差 {r["pred"]-r["diff"]:+.2f} | mf3={mf3}')

print('DONE')