# -*- coding: utf-8 -*-
"""v11.13 ranked 预测表 CSV (MAE最优: 细校准7段 + 权重重扫 + jline P95修正 + 段降权)
"""
import os, sys, numpy as np, io, pickle, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
_ALIGN = app_mod.DOMAIN_DELTA
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
with open(os.path.join(_ROOT, 'data', 'phira', 'v126_ranked_predictions.csv'), 'w', encoding='utf-8-sig', newline='') as f:  # v12.6: 新文件名
    w = csv.writer(f)
    w.writerow(['id', 'name', 'level', 'diff', 'pred', 'err', 'mf3', 'hold%', 'tags', 'kyou_type'])
    for r in ranked:
        feats = dict(r['feats'])
        lv = r['level'].upper()
        if 'AT' in lv: lv = 'AT'
        elif 'IN' in lv: lv = 'IN'
        elif 'HD' in lv: lv = 'HD'
        else: lv = 'IN'
        if lv == 'IN':
            for k, d in _ALIGN.items():
                if k in feats: feats[k] = feats[k] - d
        # v12.5: 统一调用 app.predict_from_feats (此前脚本复刻旧管线导致与网页值不一致)
        pred, _, _, _, _ = app_mod.predict_from_feats(feats, lv, is_custom=True)
        d = round(r['diff'], 1)
        ts = '+'.join(app_mod.compute_tags(feats)) if app_mod.compute_tags(feats) else '-'
        kt = app_mod.kyou_type_for(feats, r['name'], True)
        hr = r['feats'].get('hold_count', 0) / max(r['feats'].get('total_notes', 1), 1)
        w.writerow([r['id'], r['name'], r['level'], d, round(pred,2), round(pred-d,2),
                    r['feats'].get('multi_finger_3plus_events', 0), round(hr,2), ts, (kt or {}).get('type','')])
print('已保存: v1113_ranked_predictions.csv (v11.13 生产配置)')
print('DONE')
