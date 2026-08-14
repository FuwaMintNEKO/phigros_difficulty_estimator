# -*- coding: utf-8 -*-
"""unranked 分析: 按用户要求剔除 低分/少游玩/乱标难度 后, 双指vs多指预测偏差"""
import os, sys, io, json, csv, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod

# 加载预测结果
with open(os.path.join(_ROOT, 'data', 'phira', 'unranked_4star_list.csv'), encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
print('预测CSV行数:', len(rows))
meta = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'unranked_all.json'), encoding='utf-8'))
meta_by_id = {c['id']: c for c in meta}

# 用户要求: 剔除 分低(rating<阈值) 游玩少(ratingCount<阈值) 乱标难度(diff与预测差太多)
# 1) 高共识筛选
sel = []
for r in rows:
    cid = int(r['id'])
    m = meta_by_id.get(cid, {})
    rt = m.get('rating', 0) or 0
    rc = m.get('ratingCount', 0) or 0
    d = m.get('difficulty')
    try: d = float(d) if d else None
    except: d = None
    pred = float(r['pred'])
    if rt < 0.85 or rc < 100: continue
    if d is None or not (5 <= d <= 20): continue
    # 乱标难度: 谱师自标与预测差 > 2.0 (可能乱标)
    if abs(d - pred) > 2.0: continue
    sel.append({'id': cid, 'name': m.get('name',''), 'diff_self': d, 'pred': pred, 'rating': rt, 'count': rc, 'gb': float(r['gb']), 'boost': float(r['boost'])})
print(f'高共识筛选后: {len(sel)}')

# 需要特征: 从 json 提取 mf3 (选部分样本做统计)
JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star')
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
mf3s = []
sample = sel[:600]
for s in sample:
    p = os.path.join(JSON_DIR, str(s['id']) + '.json')
    try:
        with open(p, 'rb') as f:
            cd, raw = load_chart_from_bytes(f.read())
        feats = extract_features(cd, speed=1.0)
        s['mf3'] = feats.get('multi_finger_3plus_events', 0)
        s['jline_mov'] = feats.get('jline_movement_density', 0)
        s['nps'] = feats.get('real_notes_per_second', 0)
        s['dens'] = feats.get('above_avg_density_mean', 0)
    except Exception:
        s['mf3'] = -1
mf3_arr = np.array([s.get('mf3', -1) for s in sample])
ok = mf3_arr >= 0
print(f'特征提取成功: {ok.sum()}/{len(sample)}')
# 双指 vs 多指 预测 vs 自标
preds = np.array([s['pred'] for s in sample])
selfs = np.array([s['diff_self'] for s in sample])
errs = preds - selfs
for lo, hi, tag in [(0,5,'双指'), (6,29,'混合'), (30,99,'多指')]:
    mk = np.where((mf3_arr >= lo) & (mf3_arr < hi) & ok)[0]
    if len(mk):
        print(f'  {tag:<8} n={len(mk):>3} 预测-自标 bias={errs[mk].mean():+.3f} MAE={np.abs(errs[mk]).mean():.3f}')
# 高难段
print('\n=== 自标>=16.5 段 ===')
for lo, hi, tag in [(0,5,'双指'), (6,29,'混合'), (30,99,'多指')]:
    mk = np.where((mf3_arr >= lo) & (mf3_arr < hi) & ok & (selfs >= 16.5))[0]
    if len(mk):
        print(f'  {tag:<8} n={len(mk):>3} 预测-自标 bias={errs[mk].mean():+.3f}')
print('DONE')