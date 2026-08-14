# -*- coding: utf-8 -*-
"""v11.13 ranked 预测表 CSV (MAE最优: 细校准7段 + 权重重扫 + jline P95修正 + 段降权)
"""
import os, sys, numpy as np, io, pickle, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
gb, scaler = app_mod.gb, app_mod.scaler
FN, LV_ORDER = app_mod.FN, app_mod.LV_ORDER
_ALIGN = app_mod.DOMAIN_DELTA
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
with open(os.path.join(_ROOT, 'data', 'phira', 'v1113_ranked_predictions.csv'), 'w', encoding='utf-8-sig', newline='') as f:
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
        lv2 = 'IN_AT' if lv in ('IN','AT') and 'IN_AT' in LV_ORDER else lv
        if lv2 not in LV_ORDER: lv2 = LV_ORDER[-1]
        vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(lv2)] = 1.0
        x = np.array([[feats.get(n,0) for n in FN] + vec])
        p_gb = float(gb.predict(scaler.transform(x))[0])
        b, _, _ = app_mod.compute_boost(feats, 1.0, is_custom=True)
        pred = p_gb + b
        # v11.11: 堆料降权仅中段 (与 predict_one_chart 一致)
        _HIGH_TAGS = {'叠键', '多押', '变速', '位移'}
        if 14 < pred <= 16.5 and sum(1 for t in app_mod.compute_tags(feats) if t in _HIGH_TAGS) >= 2:
            pred -= b * 0.08
        act = feats.get('tracks_active_sec', 0)
        if act > 0:
            pred += 0.15*min(feats.get('tracks_4plus_sec',0)/act,0.8) + 0.55*min(feats.get('tracks_5plus_sec',0)/act,0.4) + 1.0*min(feats.get('tracks_6plus_sec',0)/act,0.15)
        hr = r['feats'].get('hold_count', 0) / max(r['feats'].get('total_notes', 1), 1)
        if hr >= 0.6: pred += 0.7
        elif hr >= 0.4: pred += 0.5
        elif hr >= 0.25: pred += 0.3
        for lo, hi, adj in app_mod._CALIB_TABLE:
            if lo < pred <= hi: pred -= adj; break
        d = round(r['diff'], 1)
        ts = '+'.join(app_mod.compute_tags(feats)) if app_mod.compute_tags(feats) else '-'
        kt = app_mod.kyou_type_for(feats, r['name'], True)
        w.writerow([r['id'], r['name'], r['level'], d, round(pred,2), round(pred-d,2),
                    r['feats'].get('multi_finger_3plus_events', 0), round(hr,2), ts, (kt or {}).get('type','')])
print('已保存: v1113_ranked_predictions.csv (v11.13 生产配置)')
print('DONE')
