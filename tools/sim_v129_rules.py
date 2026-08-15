# -*- coding: utf-8 -*-
"""v12.9 规则平衡模拟器: 多线/表演 vs 暴力多指配置
参数化复制 app.predict_from_feats 逻辑, 新规则可开关
评估: ranked(上架非整数) / 官谱bias / 锚点 / 目标谱(42113 70220 52543)
"""
import os, sys, io, pickle, json, re, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as A
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

class Opt:
    def __init__(self, **kw):
        self.rule3_excl_active = kw.get('rule3_excl_active', False)   # ③排除线活跃
        self.rule7_perf_pen = kw.get('rule7_perf_pen', 0.0)           # 表演型多指压 (负值)
        self.rule7_rot_th = kw.get('rule7_rot_th', 300.0)             # 旋转表演阈值
        self.rule7_mov_pen = kw.get('rule7_mov_pen', 0.0)             # 位移表演压 (负值)
        self.rule8_violent_lift = kw.get('rule8_violent_lift', 0.0)   # 静态暴力多指抬
        self.rule4_wmf_exempt = kw.get('rule4_wmf_exempt', False)     # ④豁免高wmf
        self.rule7c_mid_perf = kw.get('rule7c_mid_perf', 0.0)         # 中等线活跃表演压 (44705类)
        self.rule7d_static_sw = kw.get('rule7d_static_sw', 0.0)       # 静态高切换多押压 (Chart_SP类)
        self.rule1c_hispeed_sw = kw.get('rule1c_hispeed_sw', 0.0)     # 高速高切换双指抬 (Breakcore类)
        self.calib_m = kw.get('calib_m', True)

def predict_custom(feats_raw, lv, opt):
    feats = dict(feats_raw)
    lv = 'IN_AT' if lv in ('IN', 'AT') and 'IN_AT' in A.LV_ORDER else lv
    if lv not in A.LV_ORDER: lv = A.LV_ORDER[-1]
    vec = [0.0] * len(A.LV_ORDER); vec[A.LV_ORDER.index(lv)] = 1.0
    x = np.array([[feats.get(n, 0) for n in A.FN] + vec])
    p_gb = float(A.gb.predict(A.scaler.transform(x))[0])
    p_boost, dims, contribs = A.compute_boost(feats, 1.0, is_custom=True)
    p = p_gb + p_boost
    _H = {'叠键', '多押', '变速', '位移'}
    if 14 < p <= 16.5 and sum(1 for t in A.compute_tags(feats) if t in _H) >= 2:
        p -= p_boost * 0.08
    act = feats.get('tracks_active_sec', 0)
    if act > 0:
        r4 = feats.get('tracks_4plus_sec', 0) / act; r5 = feats.get('tracks_5plus_sec', 0) / act
        r6 = feats.get('tracks_6plus_sec', 0) / act; r7 = feats.get('tracks_7plus_sec', 0) / act
        p += 0.15 * min(r4, 0.8) + 0.55 * min(r5, 0.4) + 1.0 * min(r6, 0.15) + 1.6 * min(r7, 0.10)
    hr = feats.get('hold_count', 0) / max(feats.get('total_notes', 1), 1)
    if hr >= 0.6: p += 0.7
    elif hr >= 0.4: p += 0.5
    elif hr >= 0.25: p += 0.3
    mf3 = feats.get('multi_finger_3plus_events', 0); mf4 = feats.get('multi_finger_4plus_events', 0)
    bpm = feats.get('bpm', 0); odd = feats.get('odd_division_ratio', 0)
    cart = feats.get('chord_alternation_rate', 0); mov = feats.get('movement_per_second', 0)
    dens = feats.get('above_avg_density_mean', 0); ts = feats.get('type_switch_per_sec', 0)
    jmd = feats.get('jline_move_disp_per_sec', 0); jrd = feats.get('jline_rotate_disp_per_sec', 0)
    wmf = feats.get('weighted_mf_score_per_sec', 0); ns1 = feats.get('note_speed_non1_ratio', 0)
    dur = feats.get('duration_sec', 0); hr2 = feats.get('hold_ratio', 0)
    if mf3 <= 5 and dens >= 8.0 and odd >= 0.12:
        p += 0.15
    elif mf3 >= 30:
        if not (jmd >= 4.5 or jrd >= 100.0):
            p -= 0.10
    line_active = (jmd >= 4.5 or jrd >= 100.0)
    mls = feats.get('multi_line_sim_events', 0)
    if mf3 <= 15 and dens >= 10.0 and odd < 0.12 and bpm >= 170 and ts >= 0.3:
        p -= 0.55
    elif mf3 <= 15 and bpm >= 220 and ts < 0.3 and dens >= 10.0:
        p += 0.40
    elif mf3 >= 30 and cart >= 2.5 and bpm < 170 and 8.0 <= dens < 13.0:
        p += 0.55
    elif mf4 >= 50 and mov >= 60 and (not opt.rule3_excl_active or mls < 50):
        p += 0.50
    elif mf3 >= 80 and ns1 < 0.5 and dens >= 12.5 and (mf4 >= 30 or cart >= 3.8 or dens >= 15.5)             and (not opt.rule4_wmf_exempt or not (wmf >= 35.0 and mf3 >= 200)):
        p -= 0.30
    elif mf3 <= 5 and odd >= 0.12 and dens >= 12.0:
        p -= 0.35
    if p > 14.5 and dens < 8.0 and dur >= 90.0 and hr2 < 0.85:
        p -= 0.60
    if opt.rule7_perf_pen and mf3 >= 80 and dens >= 15.0 and (jrd >= opt.rule7_rot_th or jmd >= 8):
        pen = opt.rule7_perf_pen
        if jrd >= 400:
            pen = opt.rule7_perf_pen - 0.2   # 旋转表演分级: >=400 再补 -0.2
        p += pen
    if opt.rule7_mov_pen and mf3 >= 80 and jrd < opt.rule7_rot_th             and feats.get('movement_density_index', 0) >= 700 and jmd >= 4.5 and mls >= 50:
        p += opt.rule7_mov_pen
    if opt.rule8_violent_lift and wmf >= 35 and dens >= 15 and jrd < 60 and jmd < 3.5 and mf3 >= 200:
        p += opt.rule8_violent_lift
    if opt.rule7c_mid_perf and mf3 >= 80 and 7.0 <= jmd < 8.0 and jrd >= 80 and mls >= 50:
        p += opt.rule7c_mid_perf
    if opt.rule7d_static_sw and mf3 >= 80 and ts >= 1.2 and jmd < 4.5 and 10.0 <= dens < 13.0:
        p += opt.rule7d_static_sw
    if opt.rule1c_hispeed_sw and mf3 <= 5 and bpm >= 230 and ts >= 1.0 and dens >= 10.0:
        p += opt.rule1c_hispeed_sw
    if opt.calib_m:
        for _lo, _hi, _adj in A._CALIB_TABLE:
            if _lo < p <= _hi:
                p = p - _adj
                break
    return p

def lv_key(s):
    s = (s or '').upper()
    if 'AT' in s: return 'AT'
    if 'IN' in s: return 'IN'
    if 'HD' in s: return 'HD'
    return 'IN'

def feats_of(path, lv_str):
    with open(path, 'rb') as f:
        cd, raw = load_chart_from_bytes(f.read())
    feats = extract_features(cd)
    lv = lv_key(lv_str)
    if lv == 'IN':
        for k, d in A.DOMAIN_DELTA.items():
            if k in feats: feats[k] = feats[k] - d
    return feats

# ===== 数据 =====
cache = pickle.load(open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb'))
charts = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
up_ids = {x['id'] for x in charts['上架']}
ranked = [d for d in cache['ranked'] if d['id'] in up_ids]
official = cache['official']

# 目标谱
TARGETS = [
    (42113, 'Xaleid', 18.2),
    (70220, '八荒', 18.3),
    (52543, '哀煉獄歌', 18.8),
    (41242, 'Apollo', 18.0),
    (294, 'xodus#294', 17.65),
    (60137, 'Melodiniq#60137', 16.75),
    (44705, 'Xaleid#44705', 18.2),
]
def target_feats():
    out = {}
    for cid, name, tgt in TARGETS:
        p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '%d.json' % cid)
        if not os.path.exists(p):
            p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked', '%d.json' % cid)
        if os.path.exists(p):
            out[cid] = (name, tgt, feats_of(p, 'AT'), 'AT')
        else:
            for d in cache['ranked']:
                if d['id'] == cid:
                    out[cid] = (name, tgt, d['feats'], lv_key(d['level']))
    return out
TF = target_feats()

# test_charts 锚点
def parse_target(fname):
    m = re.search(r'[（(]([0-9]+(?:\.\d+)?)(?:~([0-9]+(?:\.\d+)?))?[)）]', fname)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2 if m.group(2) else float(m.group(1))
    return None
TC = []
for fn in sorted(os.listdir(os.path.join(_ROOT, 'data', 'test_charts'))):
    if not fn.endswith('.json'): continue
    t = parse_target(fn)
    if t is None: continue
    if 'Chart_SP' in fn: t = 17.65
    p = os.path.join(_ROOT, 'data', 'test_charts', fn)
    lv = 'AT' if 'AT' in fn or 'Apollo' in fn or 'Xaleid' in fn or 'Waking' in fn else 'IN'
    TC.append((fn, t, feats_of(p, lv), lv))

def run(opt):
    res = {}
    # ranked
    rows = []
    for d in ranked:
        cc = float(d['diff'])
        if cc <= 10 or abs(cc - round(cc)) < 1e-9: continue
        lv = lv_key(d['level'])
        feats = dict(d['feats'])
        if lv == 'IN':
            for k, dd in A.DOMAIN_DELTA.items():
                if k in feats: feats[k] = feats[k] - dd
        p = predict_custom(feats, lv, opt)
        rows.append((d['name'], cc, p))
    errs = np.array([r[2] - r[1] for r in rows])
    mask = np.abs(errs) < 2.0
    res['ranked'] = (int(mask.sum()), float(np.abs(errs[mask]).mean()), float(errs[mask].mean()),
                     float(np.corrcoef([r[1] for r in rows if abs(r[2]-r[1])<2], [r[2] for r in rows if abs(r[2]-r[1])<2])[0,1]))
    # 官谱 (is_custom=False: 无规则/无校准/无域对齐, 与app一致)
    oe = []
    for d in official:
        feats = dict(d['feats'])
        p = float(A.predict_from_feats(feats, lv_key(d['level']), is_custom=False)[0])
        oe.append(p - d['diff'])
    res['official_bias'] = float(np.mean(oe))
    # 目标谱
    res['targets'] = {cid: round(predict_custom(f, lv, opt), 2) for cid, (n, t, f, lv) in TF.items()}
    # test_charts 锚点
    res['tc'] = [(fn, t, round(predict_custom(f, lv, opt), 2)) for fn, t, f, lv in TC]
    return res

opts = [
    ('S0 基线', Opt()),
    ('S1 ③排除线活跃', Opt(rule3_excl_active=True)),
    ('S2 S1+旋转表演-0.8', Opt(rule3_excl_active=True, rule7_perf_pen=-0.8)),
    ('S3 S2+位移表演-0.4', Opt(rule3_excl_active=True, rule7_perf_pen=-0.8, rule7_mov_pen=-0.4)),
    ('S4 S3+静态暴力抬+0.4', Opt(rule3_excl_active=True, rule7_perf_pen=-0.8, rule7_mov_pen=-0.4, rule8_violent_lift=0.4)),
    ('S5 S4+④豁免wmf35', Opt(rule3_excl_active=True, rule7_perf_pen=-0.8, rule7_mov_pen=-0.4, rule8_violent_lift=0.4, rule4_wmf_exempt=True)),
    ('S6 仅④豁免+暴力抬', Opt(rule4_wmf_exempt=True, rule8_violent_lift=0.4)),
    ('S7 S5+M1中等表演-0.3', Opt(rule3_excl_active=True, rule7_perf_pen=-0.8, rule7_mov_pen=-0.4, rule8_violent_lift=0.4, rule4_wmf_exempt=True, rule7c_mid_perf=-0.3)),
    ('S8 S7+M2静态切换-0.4', Opt(rule3_excl_active=True, rule7_perf_pen=-0.8, rule7_mov_pen=-0.4, rule8_violent_lift=0.4, rule4_wmf_exempt=True, rule7c_mid_perf=-0.3, rule7d_static_sw=-0.4)),
    ('S9 S8+M3高速切换+0.4', Opt(rule3_excl_active=True, rule7_perf_pen=-0.8, rule7_mov_pen=-0.4, rule8_violent_lift=0.4, rule4_wmf_exempt=True, rule7c_mid_perf=-0.3, rule7d_static_sw=-0.4, rule1c_hispeed_sw=0.4)),
]
results = {}
for name, opt in opts:
    results[name] = run(opt)

print('%-28s %8s %7s %7s %7s | %s' % ('方案', 'ranked_n', 'MAE', 'bias', 'rho', '官谱bias'))
for name, r in results.items():
    n, mae, bias, rho = r['ranked']
    print('%-28s %8d %7.3f %+7.3f %7.3f | %+7.4f' % (name, n, mae, bias, rho, r['official_bias']))
print()
print('目标谱: %-22s' % '谱', ''.join('%-8s' % n.split()[0] for n, _ in opts))
for cid, (nm, t, f, lv) in TF.items():
    print('  %-20s(目标%.2f)' % (nm, t) + ''.join('%7.2f ' % results[n]['targets'][cid] for n, _ in opts))
print()
print('test_charts锚点(名/目标/各方案):')
for i, (fn, t, f, lv) in enumerate(TC):
    row = ' '.join('%5.2f' % results[n]['tc'][i][2] for n, _ in opts)
    print('  %-34s %5.2f | %s' % (fn[:34], t, row))
