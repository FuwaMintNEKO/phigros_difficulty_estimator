# -*- coding: utf-8 -*-
"""复刻v7训练路径: GB(v7残差) + v7 boost, 对全部官谱预测, 分析EZ/HD偏差来源"""
import os, sys, json, pickle, numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)
all_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    for lv in ['EZ','HD','IN','AT']:
        if lv in info['levels'] and lv in diffs:
            all_items.append({'folder':fn,'filepath':info['levels'][lv],'difficulty':diffs[lv],'level':lv})
print(f'官谱总数: {len(all_items)}')

# 加载模型
with open(os.path.join(_ROOT, 'models', '6dim_model_v7.pkl'), 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']
FLAT = m['FLAT_FEATURES']; DC = m['dynamic_cap']
print(f'GB特征数: {len(FN)}, dynamic_cap: {DC}')

def v7_boost(feats):
    raw = 0.0
    for fname, baseline, coeff in FLAT:
        val = feats.get(fname, 0)
        pv = P95.get(fname, 0)
        thresh = max(pv * 0.55, baseline * 0.5)
        if val <= thresh: continue
        excess = val / thresh - 1.0
        contrib = coeff * (excess ** 0.70)
        if val > max(P99.get(fname, 0), baseline * 0.5):
            p99_excess = val / max(P99.get(fname, 0), baseline * 0.5) - 1.0
            contrib += coeff * max(0, p99_excess) ** 0.70 * 0.5
        raw += contrib
    if raw <= DC['knee']:
        return raw
    return DC['knee'] + (raw - DC['knee']) ** DC['power']

results = []
errs = 0
for i, item in enumerate(all_items):
    try:
        cd = load_chart_json(item['filepath'])
        feats = extract_features(cd)
        if not feats:
            continue
        x = np.array([[feats.get(n, 0) for n in FN]])
        xs = scaler.transform(x)
        p_gb = float(gb.predict(xs)[0])
        p_b = v7_boost(feats)
        p_f = p_gb + p_b
        results.append({
            'name': item['folder'], 'level': item['level'], 'true': item['difficulty'],
            'gb': p_gb, 'boost': p_b, 'pred': p_f, 'err': p_f - item['difficulty'],
            'rcnps': feats.get('real_core_notes_per_second', 0),
            'density_dim': feats.get('density_dimension', 0),
            'real_active': feats.get('real_active_sec', 0),
        })
    except Exception as e:
        errs += 1
print(f'成功: {len(results)}, 失败: {errs}')

# 按难度分档
from collections import defaultdict
bands = defaultdict(list)
for r in results:
    t = r['true']
    if t < 4: band = 'EZ(<4)'
    elif t < 7: band = 'EZ(4-7)'
    elif t < 11: band = 'HD(7-11)'
    elif t < 14: band = 'IN(11-14)'
    elif t < 16.5: band = 'IN(14-16.5)'
    else: band = 'AT(>16.5)'
    bands[band].append(r)

print('\n=== 分档偏差 (v7复刻) ===')
for b, arr in bands.items():
    errs_arr = np.array([r['err'] for r in arr])
    print(f'{b:<14} n={len(arr):<4} 均值={errs_arr.mean():+.2f} 中位={np.median(errs_arr):+.2f} '
          f'p25={np.percentile(errs_arr,25):+.2f} p75={np.percentile(errs_arr,75):+.2f} '
          f'GB均值={np.mean([r["gb"] for r in arr]):.2f} boost均值={np.mean([r["boost"] for r in arr]):.2f}')

# EZ 偏差来源分析: GB vs boost 各自的贡献
print('\n=== EZ(<7) 偏差分解 ===')
ez = results[0]['true'] < 7 and results[0]['level']=='EZ'
ez_arr = [r for r in results if r['level']=='EZ']
gb_err = np.array([r['gb'] - (r['true'] - r['boost']) for r in ez_arr])
print(f'EZ图表 n={len(ez_arr)}')
print(f'  真定数均值: {np.mean([r["true"] for r in ez_arr]):.2f}')
print(f'  boost均值: {np.mean([r["boost"] for r in ez_arr]):.2f}  (残差真值=真定数-boost={np.mean([r["true"]-r["boost"] for r in ez_arr]):.2f})')
print(f'  GB预测残差均值: {np.mean([r["gb"] for r in ez_arr]):.2f}  → GB超估: {gb_err.mean():+.2f}')
print(f'  最终预测均值: {np.mean([r["pred"] for r in ez_arr]):.2f}  偏差: {np.mean([r["err"] for r in ez_arr]):+.2f}')

# 离群值影响: 排除 real_core_notes_per_second > 15 的EZ谱
ez_normal = [r for r in ez_arr if r['rcnps'] <= 15]
ez_outlier = [r for r in ez_arr if r['rcnps'] > 15]
print(f'\n  EZ中 rcnps<=15 (正常): n={len(ez_normal)}, 偏差均值={np.mean([r["err"] for r in ez_normal]):+.2f}')
if ez_outlier:
    print(f'  EZ中 rcnps>15 (离群): n={len(ez_outlier)}')
    for r in sorted(ez_outlier, key=lambda x:-x['err'])[:10]:
        print(f'    {r["name"]:<30} 真={r["true"]} GB={r["gb"]:.2f} +boost={r["boost"]:.2f} ={r["pred"]:.2f} '
              f'err={r["err"]:+.2f} rcnps={r["rcnps"]:.1f} density_dim={r["density_dim"]:.1f} real_active={r["real_active"]:.2f}')

# EZ正常谱中最大偏差Top10
print('\n=== EZ正常谱(r.cnps<=15) 最大偏差Top10 ===')
for r in sorted(ez_normal, key=lambda x:-abs(x['err']))[:10]:
    print(f'  {r["name"]:<30} 真={r["true"]} GB={r["gb"]:.2f} +boost={r["boost"]:.2f} ={r["pred"]:.2f} err={r["err"]:+.2f} rcnps={r["rcnps"]:.1f}')

# HD分析
print('\n=== HD(7-11) 偏差Top10 ===')
hd = [r for r in results if r['level']=='HD']
for r in sorted(hd, key=lambda x:-abs(x['err']))[:10]:
    print(f'  {r["name"]:<30} 真={r["true"]} GB={r["gb"]:.2f} +boost={r["boost"]:.2f} ={r["pred"]:.2f} err={r["err"]:+.2f} rcnps={r["rcnps"]:.1f}')
