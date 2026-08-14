# -*- coding: utf-8 -*-
"""v11 诊断脚本: 基线模型对上架谱 615 张的偏差诊断 (多指/双指分组)
输出: logs/exp_v11_diag.txt
"""
import os, sys, json, pickle, numpy as np
_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from feature_extractor import extract_features
from unified_parser import load_chart_from_bytes
from boost_config import MANUAL_FLAT

MODEL_PATH = os.path.join(_ROOT, 'models', '6dim_model_v10.pkl')
with open(MODEL_PATH, 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']
LV_ORDER = m.get('lv_order', ['EZ', 'HD', 'IN', 'AT'])
FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT)
CAPS = m.get('caps', {})
print(f'模型: {m.get("version")} lv_order={LV_ORDER}')

def level_key(level_str):
    s = (level_str or '').upper()
    if 'AT' in s: return 'AT'
    if 'IN' in s: return 'IN'
    if 'HD' in s: return 'HD'
    return 'IN'  # 未知自定义level(SP/ST/FM/EX等)默认IN, 与app.py一致

def level_onehot(lv):
    lv = level_key(lv)
    if 'IN_AT' in LV_ORDER and lv in ('IN', 'AT'):
        lv = 'IN_AT'
    if lv not in LV_ORDER:
        lv = LV_ORDER[-1]
    vec = [0.0] * len(LV_ORDER)
    vec[LV_ORDER.index(lv)] = 1.0
    return vec

import json as _json
_ALIGN = {}
try:
    with open(os.path.join(_ROOT, 'data', 'domain_align.json'), encoding='utf-8') as _f:
        _ALIGN = _json.load(_f).get('delta', {})
except Exception:
    pass

def _domain_align(feats, level):
    lv = level_key(level)
    if lv != 'IN':
        return feats
    for k, d in _ALIGN.items():
        if k in feats:
            feats[k] = feats[k] - d
    return feats

def predict(chart_data, level='IN'):
    feats = extract_features(chart_data, speed=1.0)
    if not feats:
        return None, None
    feats = _domain_align(feats, level)
    x = np.array([[feats.get(n, 0) for n in FN] + level_onehot(level)])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    # boost (与 app.py compute_boost 一致)
    total = 0.0
    cap_default = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0)
        pv = P95.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t: continue
        e = v / t - 1.0
        c = CAPS.get(fname, cap_default)
        if c is not None and e > c: e = c
        x_ = co * (e ** 0.70)
        p99 = max(P99.get(fname, 0), bl * 0.5)
        if v > p99:
            pe = v / p99 - 1.0
            if c is not None and pe > c: pe = c
            x_ += co * max(0, pe) ** 0.70 * 0.5
        total += x_
    return p_gb + total, feats

# ===== 加载上架谱 =====
CHART_META = os.path.join(_ROOT, 'data', 'phira', 'charts.json')
JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json')
charts = json.load(open(CHART_META, encoding='utf-8'))
meta_by_id = {}
for lst in charts.values():
    for c in lst:
        meta_by_id[c['id']] = c

# 社区定数参考: predictions.csv (diff 列)
import csv as _csv
def read_csv_cols(path):
    rows = {}
    if not os.path.exists(path): return rows
    with open(path, encoding='utf-8-sig', newline='') as f:
        rd = _csv.reader(f)
        head = next(rd)
        for c in rd:
            if len(c) < len(head): continue
            o = dict(zip(head, c))
            try:
                rows[int(o['id'])] = o
            except Exception:
                pass
    return rows
pred_old = read_csv_cols(os.path.join(_ROOT, 'data', 'phira', 'predictions.csv'))

# 邻居法参照
neigh = {}
np_path = os.path.join(_ROOT, 'data', 'phira', 'neighbor_estimate.csv')
if os.path.exists(np_path):
    with open(np_path, encoding='utf-8-sig', newline='') as f:
        rd = _csv.reader(f)
        head = next(rd)
        for c in rd:
            if len(c) < len(head): continue
            o = dict(zip(head, c))
            try:
                neigh[int(o['id'])] = float(o['neigh_est'])
            except Exception:
                pass

# ===== 批量预测 =====
results = []
for fn in sorted(os.listdir(JSON_DIR)):
    if not fn.endswith('.json'): continue
    cid = int(fn[:-5])
    meta = meta_by_id.get(cid, {})
    try:
        with open(os.path.join(JSON_DIR, fn), 'rb') as f:
            chart_data, raw_text = load_chart_from_bytes(f.read())
        if chart_data is None:
            continue
        lv = meta.get('level', 'IN')
        pred, feats = predict(chart_data, lv)
        if pred is None: continue
        results.append({
            'id': cid, 'name': meta.get('name', ''), 'level': lv,
            'diff': pred_old.get(cid, {}).get('diff'),
            'neigh': neigh.get(cid),
            'pred': pred,
            'feats': feats,
        })
    except Exception as e:
        pass
print(f'预测成功: {len(results)}')

# ===== 清洗: 社区定数 0/异常 剔除 =====
valid = [r for r in results if r['diff'] and float(r['diff']) > 10]
print(f'有效(社区定数>10): {len(valid)}')

# ===== 分段诊断 =====
def stat(rows, tag):
    bins = {}
    for r in rows:
        d = float(r['diff']); p = r['pred']
        bin_ = d < 13 and '<13' or d < 14 and '13-14' or d < 15 and '14-15' or d < 16 and '15-16' or d < 17 and '16-17' or '>=17'
        b = bins.setdefault(bin_, {'n': 0, 'pb': 0, 'pmae': 0, 'nb': 0, 'nmae': 0, 'mf3': [], 'eff': [], 'wb': 0})
        b['n'] += 1
        b['pb'] += p - d
        b['pmae'] += abs(p - d)
        f = r['feats']
        b['mf3'].append(f.get('multi_finger_3plus_events', 0))
        b['eff'].append(f.get('eff_peak_tps_1s', 0))
        if r['neigh']:
            b['nb'] += p - r['neigh']
            b['nmae'] += abs(p - r['neigh'])
    print(f'\n=== {tag} ===')
    print('分段 | n | pred-社区 | MAE | pred-邻居 | MAE(neigh) | mf3均值 | eff峰值均值')
    for k in sorted(bins, key=lambda x: float(x.replace('<', '0').replace('-', '.').replace('>=', '99'))):
        b = bins[k]
        mf3 = np.mean(b['mf3']) if b['mf3'] else 0
        eff = np.mean(b['eff']) if b['eff'] else 0
        nb = b['nb'] / b['n'] if b['n'] else 0
        nmae = b['nmae'] / b['n'] if b['n'] else 0
        print(f'  {k}: n={b["n"]} | {b["pb"]/b["n"]:+.3f} | {b["pmae"]/b["n"]:.3f} | {nb:+.3f} | {nmae:.3f} | {mf3:.1f} | {eff:.1f}')

stat(valid, '上架谱 基线模型')

# ===== 多指 vs 双指分组 (16+ 段) =====
print('\n=== 16+ 段: 多指/双指分组 (pred - 社区diff) ===')
hi = [r for r in valid if float(r['diff']) >= 16]
def mf_group(r):
    f = r['feats']
    mf3 = f.get('multi_finger_3plus_events', 0)
    eff = f.get('eff_peak_tps_1s', 0)
    # 多指谱: mf3 高且 eff 低 (多指全押); 双指谱: mf3 低
    if mf3 >= 30: return '多指(mf3>=30)'
    if mf3 <= 5: return '双指(mf3<=5)'
    return '混合'
groups = {}
for r in hi:
    g = mf_group(r)
    gr = groups.setdefault(g, {'n': 0, 'b': 0, 'mae': 0, 'nb': 0})
    d = float(r['diff'])
    gr['n'] += 1; gr['b'] += r['pred'] - d; gr['mae'] += abs(r['pred'] - d)
    if r['neigh']: gr['nb'] += r['pred'] - r['neigh']
for g, gr in groups.items():
    nb = gr['nb'] / gr['n'] if gr['n'] else 0
    print(f'  {g}: n={gr["n"]} | pred-社区 {gr["b"]/gr["n"]:+.3f} | MAE {gr["mae"]/gr["n"]:.3f} | pred-邻居 {nb:+.3f}')

# ===== 外推段排序一致性 (pred vs 社区 diff, >=17.7) =====
print('\n=== 外推段 (社区>=17.7) 排序一致性 ===')
ext = [r for r in valid if float(r['diff']) >= 17.7]
if len(ext) >= 5:
    from scipy.stats import spearmanr
    d_arr = np.array([float(r['diff']) for r in ext])
    p_arr = np.array([r['pred'] for r in ext])
    rho, pv = spearmanr(d_arr, p_arr)
    print(f'  n={len(ext)} Spearman rho={rho:.3f} (p={pv:.4f})')
    for r in sorted(ext, key=lambda x: -float(x['diff'])):
        print(f'    {r["name"][:24]}: 社区 {float(r["diff"]):.1f} | 预测 {r["pred"]:.2f} | 差 {r["pred"]-float(r["diff"]):+.2f}')
else:
    print(f'  n={len(ext)} 不足5, 跳过')
print('\nDONE')