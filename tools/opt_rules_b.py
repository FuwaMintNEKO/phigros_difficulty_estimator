# -*- coding: utf-8 -*-
"""B计划: 规则力度全局联合调参 (坐标下降)
- 预计算: 每谱 base(GB+boost+堆料+定轨+hold) + 规则命中向量 (类型规则均为加性力度)
- 参数: 16条类型规则力度 + 7段校准 = 23个; 阈值保持不动
- 目标: ranked MAE(上架非整数, 排除乱标) + λ×锚点超出±0.28的平方惩罚
- 坐标下降 3轮 × 23参数 × 21网格点; 参数范围 ±0.25(校准±0.15)
"""
import os, sys, io, pickle, json, re, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as A
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

# ===== 参数定义: (名称, 当前值, 范围半径) — 与app.py v12.14同步 =====
PARAMS = [
    ('df_base',  -0.08, 0.25),   # 双指底力
    ('static_mf', -0.07, 0.25),  # 单面静态多押压
    ('r1',  -0.48, 0.25),  ('r1b', 0.25, 0.25),  ('r1c', 0.35, 0.25),
    ('r2',   0.40, 0.25),  ('r3',  0.50, 0.25),  ('r4', -0.30, 0.25),
    ('r5',  -0.35, 0.25),  ('r6', -0.35, 0.25),
    ('r7a', -0.80, 0.25),  ('r7a400', -0.20, 0.25),  ('r7b', -0.40, 0.25),
    ('r8',   0.40, 0.25),  ('r9',  -0.30, 0.25),  ('r10', -0.40, 0.25),
    ('r11',  0.40, 0.25),   # 暴力高密度键盘抬
    ('cal_12_13', 0.00, 0.15), ('cal_13_14', 0.05, 0.15),   # 低段校准解锁(v12.15最终优化)
    ('cal_14_15', -0.05, 0.15), ('cal_15_16', 0.04, 0.15),
    ('cal_16_165', 0.40, 0.15), ('cal_165_17', 0.07, 0.15), ('cal_17_99', -0.09, 0.15),
]
CUR = np.array([p[1] for p in PARAMS])
RAD = np.array([p[2] for p in PARAMS])
CAL_EDGES = [(12,13),(13,14),(14,15),(15,16),(16,16.5),(16.5,17),(17,99)]

# ===== 数据 =====
cache = pickle.load(open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb'))
charts = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
up_ids = {x['id'] for x in charts['上架']}

def lv_key(s):
    s = (s or '').upper()
    if 'AT' in s: return 'AT'
    if 'IN' in s: return 'IN'
    if 'HD' in s: return 'HD'
    return 'IN'

# ranked 样本
R = []
for d in cache['ranked']:
    if d['id'] not in up_ids: continue
    cc = float(d['diff'])
    if cc <= 10 or abs(cc - round(cc)) < 1e-9: continue
    lv = lv_key(d['level'])
    feats = dict(d['feats'])
    if lv == 'IN':
        for k, dd in A.DOMAIN_DELTA.items():
            if k in feats: feats[k] = feats[k] - dd
    R.append((d['name'], cc, feats))
print('ranked样本: %d' % len(R))

# 锚点
def feats_of(path, lv):
    cd, raw = load_chart_from_bytes(open(path, 'rb').read())
    fe = extract_features(cd)
    if lv == 'IN':
        for k, dd in A.DOMAIN_DELTA.items():
            if k in feats: feats[k] = feats[k] - dd
    return fe

ANCH = []
def parse_target(fname):
    m = re.search(r'[（(]([0-9]+(?:[.][0-9]+)?)(?:~([0-9]+(?:[.][0-9]+)?))?[)）]', fname)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2 if m.group(2) else float(m.group(1))
    return None
for fn in sorted(os.listdir(os.path.join(_ROOT, 'data', 'test_charts'))):
    if not fn.endswith('.json'): continue
    t = parse_target(fn)
    if t is None or t < 14.5: continue   # 低段锚点按用户要求不修
    if 'Lemegeton' in fn: continue       # 社区定数不可信(用户确认), 移除
    if 'Chart_SP' in fn: t = 17.65
    if 'Runengon' in fn: t = 16.8
    if 'おぎゃり' in fn: t = 16.5
    p = os.path.join(_ROOT, 'data', 'test_charts', fn)
    lv = 'AT' if ('AT' in fn or 'Apollo' in fn or 'Xaleid' in fn or 'Waking' in fn or 'Final' in fn or 'ギザバ' in fn) else 'IN'
    ANCH.append((fn[:26], t, feats_of(p, lv)))
ANCH_IDS = [(41242, 'Apollo41242', 18.0, 'AT'), (294, 'xodus294', 17.65, 'AT'),
            (60137, 'Melodiniq60137', 16.75, 'AT'), (44705, 'Xaleid44705', 18.2, 'AT'),
            (42113, 'Xaleid42113', 18.2, 'AT'), (70220, '八荒', 18.3, 'AT'), (52543, '哀煉獄歌', 18.9, 'AT')]
for cid, nm, tgt, lv in ANCH_IDS:
    p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '%d.json' % cid)
    if not os.path.exists(p):
        p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked', '%d.json' % cid)
    if os.path.exists(p):
        ANCH.append((nm, tgt, feats_of(p, lv)))
print('锚点样本: %d' % len(ANCH))

# ===== 预计算 base + hit 向量 =====
def base_and_hits(feats):
    """返回 (base, hits列表, boost, 堆料是否触发, p0用于⑥判定) — 不含类型规则/校准"""
    lv = 'IN_AT'
    vec = [0.0] * len(A.LV_ORDER); vec[A.LV_ORDER.index('IN_AT')] = 1.0
    x = np.array([[feats.get(n, 0) for n in A.FN] + vec])
    p_gb = float(A.gb.predict(A.scaler.transform(x))[0])
    p_boost, dims, contribs = A.compute_boost(feats, 1.0, is_custom=True)
    p = p_gb + p_boost
    _H = {'叠键', '多押', '变速', '位移'}
    stack = 0.0
    if 14 < p <= 16.5 and sum(1 for t in A.compute_tags(feats) if t in _H) >= 2:
        stack = -p_boost * 0.08
    act = feats.get('tracks_active_sec', 0)
    tr = 0.0
    if act > 0:
        r4 = feats.get('tracks_4plus_sec', 0) / act; r5 = feats.get('tracks_5plus_sec', 0) / act
        r6 = feats.get('tracks_6plus_sec', 0) / act; r7 = feats.get('tracks_7plus_sec', 0) / act
        tr = 0.15 * min(r4, 0.8) + 0.55 * min(r5, 0.4) + 1.0 * min(r6, 0.15) + 1.6 * min(r7, 0.10)
    hr = feats.get('hold_count', 0) / max(feats.get('total_notes', 1), 1)
    hh = 0.7 if hr >= 0.6 else (0.5 if hr >= 0.4 else (0.3 if hr >= 0.25 else 0.0))
    base = p + stack + tr + hh
    mf3 = feats.get('multi_finger_3plus_events', 0); mf4 = feats.get('multi_finger_4plus_events', 0)
    bpm = feats.get('bpm', 0); odd = feats.get('odd_division_ratio', 0)
    cart = feats.get('chord_alternation_rate', 0); mov = feats.get('movement_per_second', 0)
    dens = feats.get('above_avg_density_mean', 0); ts = feats.get('type_switch_per_sec', 0)
    jmd = feats.get('jline_move_disp_per_sec', 0); jrd = feats.get('jline_rotate_disp_per_sec', 0)
    wmf = feats.get('weighted_mf_score_per_sec', 0); ns1 = feats.get('note_speed_non1_ratio', 0)
    mls = feats.get('multi_line_sim_events', 0)
    hits = {}
    hits['df_base'] = 1.0 if (mf3 <= 5 and dens >= 8.0 and odd >= 0.12) else 0.0
    hits['static_mf'] = 1.0 if (mf3 >= 30 and not (jmd >= 4.5 or jrd >= 100.0)) else 0.0
    # ①~⑤ elif链
    chain = None
    if mf3 <= 15 and dens >= 10.0 and odd < 0.12 and 170.0 <= bpm < 250.0 and ts >= 0.3: chain = 'r1'
    elif mf3 <= 15 and bpm >= 220 and ts < 0.3 and dens >= 10.0: chain = 'r1b'
    elif mf3 >= 30 and cart >= 2.5 and bpm < 170 and 8.0 <= dens < 13.0: chain = 'r2'
    elif mf4 >= 50 and mov >= 60 and mls < 50: chain = 'r3'
    elif mf3 >= 80 and ns1 < 0.5 and dens >= 12.5 and (mf4 >= 30 or cart >= 3.8 or dens >= 15.5)             and not (wmf >= 35.0 and mf3 >= 200): chain = 'r4'
    elif mf3 <= 5 and odd >= 0.12 and dens >= 12.0: chain = 'r5'
    for k in ['r1', 'r1b', 'r2', 'r3', 'r4', 'r5']:
        hits[k] = 1.0 if chain == k else 0.0
    hits['r1c'] = 1.0 if (mf3 <= 5 and bpm >= 230 and ts >= 1.0 and dens >= 10.0) else 0.0
    p0 = base + sum(CUR[PARAMS.index((k,))] if False else 0 for k in [])  # placeholder
    hits['r6'] = 1.0 if (base > 14.5 and dens < 8.0 and feats.get('duration_sec', 0) >= 90.0 and feats.get('hold_ratio', 0) < 0.85) else 0.0
    hits['r7a'] = 1.0 if (mf3 >= 80 and dens >= 15.0 and (jrd >= 300.0 or jmd >= 8.0)) else 0.0
    hits['r7a400'] = 1.0 if (hits['r7a'] and jrd >= 400.0) else 0.0
    hits['r7b'] = 1.0 if (mf3 >= 80 and jrd < 300.0 and feats.get('movement_density_index', 0) >= 700 and jmd >= 4.5 and mls >= 50) else 0.0
    hits['r8'] = 1.0 if (wmf >= 35.0 and dens >= 15.0 and jrd < 60.0 and jmd < 3.5 and mf3 >= 200) else 0.0
    hits['r9'] = 1.0 if (mf3 >= 80 and 7.0 <= jmd < 8.0 and jrd >= 80.0 and mls >= 50) else 0.0
    hits['r10'] = 1.0 if (mf3 >= 80 and ts >= 1.2 and jmd < 4.5 and 10.0 <= dens < 13.0) else 0.0
    hits['r11'] = 1.0 if (bpm >= 250.0 and feats.get('eff_peak_tps_1s', 0) >= 32.0 and dens >= 15.0
                          and jmd < 4.0 and jrd < 60.0 and mf3 >= 50
                          and not (wmf >= 35.0 and mf3 >= 200)) else 0.0
    return base, hits

names = [p[0] for p in PARAMS]
RB = []
for nm, cc, f in R:
    base, hits = base_and_hits(f)
    RB.append((nm, cc, base, np.array([hits.get(k, 0.0) for k in names])))
AB = []
for nm, tgt, f in ANCH:
    base, hits = base_and_hits(f)
    AB.append((nm, tgt, base, np.array([hits.get(k, 0.0) for k in names])))

RB_base = np.array([r[2] for r in RB]); RB_hit = np.array([r[3] for r in RB])
RB_diff = np.array([r[1] for r in RB])
AB_base = np.array([a[2] for a in AB]); AB_hit = np.array([a[3] for a in AB])
AB_tgt = np.array([a[1] for a in AB])

COMM_BINS = []
try:
    _cc = json.load(open(os.path.join(_ROOT, 'data', 'community_calib.json'), encoding='utf-8')).get('bins', {})
    for _k, _v in _cc.items():
        _lo_s, _hi_s = _k.split('-')
        COMM_BINS.append((float(_lo_s), float(_hi_s), float(_v.get('adj', 0.0))))
except Exception:
    pass

def predict_all(params):
    pr = RB_base + RB_hit @ params
    pa = AB_base + AB_hit @ params
    # 手工校准(分段, 基于含规则后的预测值; (lo,hi]与app一致)
    for lo, hi, ki in [(12,13,'cal_12_13'),(13,14,'cal_13_14'),(14,15,'cal_14_15'),
                       (15,16,'cal_15_16'),(16,16.5,'cal_16_165'),(16.5,17,'cal_165_17'),(17,99,'cal_17_99')]:
        j = names.index(ki)
        mr = (pr > lo) & (pr <= hi)
        ma = (pa > lo) & (pa <= hi)
        pr[mr] -= params[j]
        pa[ma] -= params[j]
    # 社区校准层 ([lo,hi)与app一致)
    for lo, hi, adj in COMM_BINS:
        mr = (pr >= lo) & (pr < hi)
        ma = (pa >= lo) & (pa < hi)
        pr[mr] += adj
        pa[ma] += adj
    return pr, pa

def objective(params, lam=200.0):
    pr, pa = predict_all(params)
    err = pr - RB_diff
    mask = np.abs(err) < 2.0
    mae = np.abs(err[mask]).mean()
    aerr = pa - AB_tgt
    pen = np.maximum(np.abs(aerr) - 0.28, 0.0) ** 2
    return mae + lam * pen.mean()

# 基线
p0, a0 = predict_all(CUR)
err0 = p0 - RB_diff
mask0 = np.abs(err0) < 2.0
print('基线: MAE=%.4f  锚点超界平方均值=%.5f  目标=%.5f' % (
    np.abs(err0[mask0]).mean(), np.maximum(np.abs(a0 - AB_tgt) - 0.28, 0).mean(), objective(CUR)))
aerr0 = a0 - AB_tgt
for i, (nm, tgt, _) in enumerate(ANCH):
    if abs(aerr0[i]) > 0.28:
        print('  锚点超界: %-26s tgt=%5.2f pred=%5.2f err=%+.2f' % (nm, tgt, a0[i], aerr0[i]))

# ===== 坐标下降 =====
params = CUR.copy()
best_obj = objective(params)
print('初始目标: %.5f' % best_obj)
for rnd in range(3):
    improved = 0
    for pi in range(len(names)):
        lo = CUR[pi] - RAD[pi]
        hi = CUR[pi] + RAD[pi]
        grid = np.linspace(lo, hi, 21)
        objs = []
        for v in grid:
            t = params.copy(); t[pi] = v
            objs.append(objective(t))
        k = int(np.argmin(objs))
        if objs[k] < best_obj - 1e-7:
            params[pi] = grid[k]
            best_obj = objs[k]
            improved += 1
    print('轮%d: 目标=%.5f, 改进参数%d个' % (rnd + 1, best_obj, improved))
    if improved == 0:
        break

pr, pa = predict_all(params)
err = pr - RB_diff
mask = np.abs(err) < 2.0
print()
print('=== 优化结果 ===')
print('ranked MAE: %.4f -> %.4f' % (np.abs(err0[mask0]).mean(), np.abs(err[mask]).mean()))
print('ranked bias: %+.4f -> %+.4f' % (err0[mask0].mean(), err[mask].mean()))
print('锚点超界数: %d -> %d' % (int((np.abs(a0-AB_tgt)>0.28).sum()), int((np.abs(pa-AB_tgt)>0.28).sum())))
print()
print('参数变化:')
for i, nm in enumerate(names):
    if abs(params[i] - CUR[i]) > 0.015:
        print('  %-12s %+.2f -> %+.2f' % (nm, CUR[i], params[i]))
print()
print('锚点明细:')
for i, (nm, tgt, _, _) in enumerate(AB):
    flag = ' <<<' if abs(pa[i]-tgt) > 0.28 else ''
    print('  %-26s tgt=%5.2f 基线%5.2f 优化%5.2f%s' % (nm, tgt, a0[i], pa[i], flag))
