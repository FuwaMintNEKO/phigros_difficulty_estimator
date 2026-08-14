# -*- coding: utf-8 -*-
"""全面分析当前模型偏差:
1. 官方谱: 整体/按level/按定数桶 的 signed 偏差 (正=高估, 负=低估)
2. 低估/高估最严重的官方谱 top
3. 低估组 vs 高估组 的特征对比 (找出系统性盲区)
"""
import os, sys, pickle
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import numpy as np
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
import app

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
LEVELS = ['EZ', 'HD', 'IN', 'AT']

charts = find_chart_files(CHART_DIR)
diffs = load_difficulty_tsv(TSV)

rows = []
for folder, info in charts.items():
    sid = info['song_id']
    for level in LEVELS:
        fp = info['levels'].get(level)
        if not fp:
            continue
        d = (diffs.get(sid) or {}).get(level)
        if d is None:
            continue
        try:
            with open(fp, 'rb') as f:
                cd, _ = load_chart_from_bytes(f.read())
            if cd is None:
                continue
            res, err = app.predict_one_chart(cd, speed=1.0, level=level)
            if res is None:
                print(f'PRED_ERR {sid} {level}: {err}')
                continue
            fe = extract_features(cd)
            rows.append({'song': sid, 'level': level, 'true': d, 'pred': res['prediction'],
                         'gb': res['gb'], 'boost': res['boost'], 'feats': fe})
        except Exception as e:
            print(f'ERR {sid} {level}: {e}')
    if len(rows) > 0 and len(rows) % 200 == 0:
        print(f'...已加载 {len(rows)}')

print(f'官方谱样本: {len(rows)}')
print(f'模型: {app.VERSION}')

# 整体
all_err = [r['pred'] - r['true'] for r in rows]
print(f'\n整体 MAE={np.mean(np.abs(all_err)):.4f}  有符号偏差={np.mean(all_err):+.4f}')

# 按 level
print('\n===== 按 level =====')
for lv in LEVELS:
    rs = [r for r in rows if r['level'] == lv]
    if not rs:
        continue
    errs = [r['pred'] - r['true'] for r in rs]
    print(f'{lv:>3} n={len(rs):>4}  MAE={np.mean(np.abs(errs)):.4f}  有符号={np.mean(errs):+.4f}')

# 按定数桶
print('\n===== 按真实定数桶 =====')
bins = [(0, 8), (8, 10), (10, 12), (12, 13.5), (13.5, 15), (15, 16.5), (16.5, 99)]
for lo, hi in bins:
    rs = [r for r in rows if lo <= r['true'] < hi]
    if not rs:
        continue
    errs = [r['pred'] - r['true'] for r in rs]
    print(f'[{lo:>4},{hi:>4}) n={len(rs):>4}  MAE={np.mean(np.abs(errs)):.4f}  有符号={np.mean(errs):+.4f}')

# 低估/高估 top15
print('\n===== 低估最严重 top15 (预测过低) =====')
for r in sorted(rows, key=lambda x: x['pred'] - x['true'])[:15]:
    print(f'{r["song"][:38]:<38} {r["level"]:>3} 真={r["true"]:>5.1f} 预测={r["pred"]:>5.1f} 差={r["pred"]-r["true"]:+.2f}')
print('\n===== 高估最严重 top15 (预测过高) =====')
for r in sorted(rows, key=lambda x: x['pred'] - x['true'], reverse=True)[:15]:
    print(f'{r["song"][:38]:<38} {r["level"]:>3} 真={r["true"]:>5.1f} 预测={r["pred"]:>5.1f} 差={r["pred"]-r["true"]:+.2f}')

# 特征对比: 低估组(<-0.8) vs 高估组(>+0.8)
print('\n===== 特征对比: 低估组 vs 高估组 (官方谱, 偏差绝对值>0.8) =====')
under = [r for r in rows if r['pred'] - r['true'] < -0.8]
over = [r for r in rows if r['pred'] - r['true'] > 0.8]
print(f'低估组 n={len(under)}, 高估组 n={len(over)}')
KEY_F = ['real_core_notes_per_second', 'above_avg_density_mean', 'weighted_mf_score_per_sec',
         'chord_alternation_rate', 'type_switch_per_sec', 'above_avg_duration_sec',
         'tempo_change_count', 'speed_volatility', 'jline_movement_density',
         'jline_rotate_density', 'hold_interference_index', 'density_transition_std',
         'chord_size_entropy', 'note_clutter_ratio', 'pattern_switch_rate',
         'rhythm_entropy', 'direction_irregularity', 'stair_complexity',
         'note_speed_non1_ratio', 'chord_jack_density']
print(f'{"特征":<28} {"低估组均值":>10} {"高估组均值":>10} {"差异":>8}')
for kf in KEY_F:
    u = np.mean([r['feats'].get(kf, 0) for r in under])
    o = np.mean([r['feats'].get(kf, 0) for r in over])
    print(f'{kf:<28} {u:>10.3f} {o:>10.3f} {u-o:>+8.3f}')

# 保存行数据供后续分析
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_bias_rows.pkl'), 'wb') as f:
    pickle.dump([{k: v for k, v in r.items() if k != 'feats'} | {'feats': r['feats']} for r in rows], f)
print('\n已保存 _bias_rows.pkl')
