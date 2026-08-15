# -*- coding: utf-8 -*-
"""社区定数校准层原型: 分段(0.5档)对齐社区非整数定数
- 样本: 上架 + 非整数定数 + |err|<2 (排除乱标)
- 每段 n>=5 才校准; 校准量 = -mean_err, clip ±0.3
- 高段特调: 17+ 段系数 0.5 (社区定数膨胀, 半对齐)
- 输出: 校准表 + 验证(ranked MAE / 官谱bias / 锚点)
"""
import os, sys, io, pickle, json, numpy as np, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as A
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

cache = pickle.load(open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb'))
charts = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
up_ids = {x['id'] for x in charts['上架']}

def lv_key(s):
    s = (s or '').upper()
    if 'AT' in s: return 'AT'
    if 'IN' in s: return 'IN'
    if 'HD' in s: return 'HD'
    return 'IN'

rows = []
for d in cache['ranked']:
    if d['id'] not in up_ids: continue
    cc = float(d['diff'])
    if cc <= 10 or abs(cc - round(cc)) < 1e-9: continue
    lv = lv_key(d['level'])
    feats = dict(d['feats'])
    if lv == 'IN':
        for k, dd in A.DOMAIN_DELTA.items():
            if k in feats: feats[k] = feats[k] - dd
    p = float(A.predict_from_feats(feats, lv, True)[0])
    if abs(p - cc) < 2.0:
        rows.append((d['name'], cc, p, p - cc, d['id'], feats))
print('样本(非整数+排除乱标): %d' % len(rows))

# 分段 err 均值 (按预测值 p 分段, 0.5档)
edges = list(np.arange(11.0, 20.0, 0.5))
calib = {}
print()
print('=== 分段统计 ===')
for lo in edges:
    hi = lo + 0.5
    sel = [r for r in rows if lo <= r[2] < hi]
    if len(sel) < 5:
        print('  %4.1f-%4.1f: n=%2d 跳过' % (lo, hi, len(sel)))
        continue
    errs = np.array([r[3] for r in sel])
    # 分段系数: 低段0.5(社区低段定数乱) / 中段1.0 / 16.0-17.0锚点密集区0(社区低标,用户锚点已修正) / 17+段0.5(社区膨胀半对齐)
    if 16.0 <= lo < 17.0:
        coef = 0.0
    elif lo < 14.0:
        coef = 0.5
    elif lo >= 17.0:
        coef = 0.5
    else:
        coef = 1.0
    adj = float(np.clip(-errs.mean() * coef, -0.3, 0.3))
    calib['%.1f-%.1f' % (lo, hi)] = {'n': len(sel), 'mean_err': round(float(errs.mean()), 3), 'adj': round(adj, 3)}
    print('  %4.1f-%4.1f: n=%3d mean_err=%+.3f -> adj=%+.3f (coef=%.1f)' % (lo, hi, len(sel), errs.mean(), adj, coef))

def apply_calib(p):
    for k, v in calib.items():
        lo_s, hi_s = k.split('-')
        lo, hi = float(lo_s), float(hi_s)
        if lo <= p < hi:
            return p + v['adj']
    return p

# 验证 ranked
errs_new = []
for nm, cc, p, e, cid, f in rows:
    errs_new.append(apply_calib(p) - cc)
errs_new = np.array(errs_new)
errs_old = np.array([r[3] for r in rows])
print()
print('ranked MAE: %.3f -> %.3f' % (np.abs(errs_old).mean(), np.abs(errs_new).mean()))
print('ranked bias: %+.3f -> %+.3f' % (errs_old.mean(), errs_new.mean()))
print('rho: %.3f -> %.3f' % (
    np.corrcoef([r[1] for r in rows], [r[2] for r in rows])[0,1],
    np.corrcoef([r[1] for r in rows], [apply_calib(r[2]) for r in rows])[0,1]))

# 锚点验证 (test_charts + id)
def feats_of(path, lv):
    cd, raw = load_chart_from_bytes(open(path, 'rb').read())
    fe = extract_features(cd)
    if lv == 'IN':
        for k, dd in A.DOMAIN_DELTA.items():
            if k in feats: feats[k] = feats[k] - dd
    return fe

def parse_target(fname):
    m = re.search(r'[（(]([0-9]+(?:[.][0-9]+)?)(?:~([0-9]+(?:[.][0-9]+)?))?[)）]', fname)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2 if m.group(2) else float(m.group(1))
    return None

ANCH = []
for fn in sorted(os.listdir(os.path.join(_ROOT, 'data', 'test_charts'))):
    if not fn.endswith('.json'): continue
    t = parse_target(fn)
    if t is None or t < 14.5: continue
    if 'Lemegeton' in fn: continue
    if 'Chart_SP' in fn: t = 17.65
    if 'Runengon' in fn: t = 16.8
    if 'おぎゃり' in fn: t = 16.5
    p = os.path.join(_ROOT, 'data', 'test_charts', fn)
    lv = 'AT' if ('AT' in fn or 'Apollo' in fn or 'Xaleid' in fn or 'Waking' in fn or 'Final' in fn or 'ギザバ' in fn) else 'IN'
    ANCH.append((fn[:26], t, feats_of(p, lv), lv))
for cid, nm, tgt in [(41242,'Apollo41242',18.0),(294,'xodus294',17.65),(60137,'Melodiniq60137',16.75),(44705,'Xaleid44705',18.2),(42113,'Xaleid42113',18.2),(70220,'八荒',18.3),(52543,'哀',18.9)]:
    p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '%d.json' % cid)
    if not os.path.exists(p):
        p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked', '%d.json' % cid)
    if os.path.exists(p):
        ANCH.append((nm, tgt, feats_of(p, 'AT'), 'AT'))
print()
print('=== 锚点 (校准前 -> 校准后) ===')
bad = 0
for nm, tgt, f, lv in ANCH:
    p0 = float(A.predict_from_feats(f, lv, True)[0])
    p1 = apply_calib(p0)
    flag = '' if abs(p1 - tgt) <= 0.28 else ' <<<'
    if flag: bad += 1
    print('  %-24s tgt=%5.2f  %.2f -> %.2f%s' % (nm, tgt, p0, p1, flag))
print('超界: %d' % bad)

# 保存
with open(os.path.join(_ROOT, 'data', 'community_calib.json'), 'w', encoding='utf-8') as f:
    json.dump({'version': 'v12.12proto', 'bins': calib}, f, ensure_ascii=False, indent=1)
print()
print('校准表已保存: data/community_calib.json')
