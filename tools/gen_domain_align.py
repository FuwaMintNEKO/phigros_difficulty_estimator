# -*- coding: utf-8 -*-
"""生成密度域对齐数据 data/domain_align.json
delta[feat] = 自制IN(14-16.5)均值 - 官谱IN(14-16.5)均值
(预测时对自制谱 IN 段特征减去 delta, 即向官谱分布对齐)
"""
import os, sys, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from unified_parser import load_chart_from_bytes
import app

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json')

def dens_feats():
    feats = sorted(n for n in app.FN if any(k in n for k in
        ['density_16beat', 'density_8beat', 'density_4beat', 'density_2beat', 'density_1beat',
         'density_0.5beat', 'density_0.25beat', 'notes_per_second', 'notes_per_beat',
         'tap_burst_', 'peak_density', 'rcnps', 'core_peak', 'burst_intensity',
         'fast_note_density', 'micro_peak', 'high_density_ratio', 'density_spike_ratio',
         'p90_density', 'p75_density', 'mean_density', 'core_std_density',
         'multi_finger_density', 'weighted_mf_score_per_sec', 'jline_movement_density']))
    # V10.1 特征大改新增: 有效单指密度 / 对拍对切 / 位移复合 (自制谱系统性偏高, 需对齐)
    feats += sorted(n for n in app.FN if any(k in n for k in
        ['eff_peak', 'eff_avg', 'chord_complexity', 'chord_chord_alt', 'chord_entropy_norm',
         'movement_per_second', 'movement_density']))
    return sorted(set(feats))

D = dens_feats()
print(f'密度类特征 {len(D)} 个')

charts = find_chart_files(CHART_DIR)
diffs = load_difficulty_tsv(TSV)
off = []
for fn, info in charts.items():
    if 'IN' not in info['levels']:
        continue
    d = (diffs.get(info['song_id']) or {}).get('IN')
    if d is None or not (14.0 <= d < 16.5):
        continue
    try:
        f = extract_features(load_chart_json(info['levels']['IN']))
        if f:
            off.append(f)
    except Exception:
        pass

meta = {}
charts_meta = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
for lst in charts_meta.values():
    for c in lst:
        meta[c['id']] = c
selfl = []
for fn in sorted(os.listdir(JSON_DIR)):
    if not fn.endswith('.json'):
        continue
    info = meta.get(int(fn[:-5]), {})
    diff = info.get('difficulty', 0)
    if not (14.0 <= diff < 16.5):
        continue
    try:
        with open(os.path.join(JSON_DIR, fn), 'rb') as f:
            raw = f.read()
        cd, _ = load_chart_from_bytes(raw)
        if cd is None:
            continue
        feats = extract_features(cd)
        if feats:
            selfl.append(feats)
    except Exception:
        pass

print(f'官谱 IN(14-16.5): {len(off)}, 自制 IN(14-16.5): {len(selfl)}')
off_mean = {k: float(np.mean([f.get(k, 0) for f in off])) for k in D}
self_mean = {k: float(np.mean([f.get(k, 0) for f in selfl])) for k in D}
delta = {k: round(self_mean[k] - off_mean[k], 6) for k in D}

out = {'features': D, 'delta': delta, 'source': 'official IN(14-16.5) vs custom IN(14-16.5)'}
out_path = os.path.join(_ROOT, 'data', 'domain_align.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f'已保存 {out_path}')
for k in D:
    print(f'  {k:<34} delta={delta[k]:+.4f}')
