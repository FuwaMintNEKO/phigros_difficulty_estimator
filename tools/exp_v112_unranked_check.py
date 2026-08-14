# -*- coding: utf-8 -*-
"""v11.2 实验 t3: 未上架957谱预测分布与官谱域对齐检查

对 data/phira/json_unranked/*.json (957张, RPE/PE格式) 提取特征并预测:
  - unified_parser.load_chart_from_bytes + feature_extractor.extract_features (speed=1.0)
  - 6dim_model_v11_1.pkl 预测 (GB + 条件boost mf3衰减/eff抬升 + 校准, 不含定轨加成)
统计:
  - 预测分布 (均值/分位数/>18占比/>19占比), 与社区diff (unranked_final_download.json) 的偏差
  - 特征域偏移: above_avg_density_mean/eff_avg/mf3 等 vs 官谱 16+ 段 P90/P99

输出: logs/exp_v112_unranked_check.txt + tools/notes_v112_unranked.md
"""
import os, sys, json, pickle, time
import numpy as np
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT as _MANUAL_FLAT_BOOT
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

MODEL_PATH = os.path.join(_ROOT, 'models', '6dim_model_v11_1.pkl')
CACHE_PATH = os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl')
ALIGN_PATH = os.path.join(_ROOT, 'data', 'domain_align.json')
LIST_PATH = os.path.join(_ROOT, 'data', 'phira', 'unranked_final_download.json')
JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked')
OUT_TXT = os.path.join(_ROOT, 'logs', 'exp_v112_unranked_check.txt')
OUT_MD = os.path.join(_ROOT, 'tools', 'notes_v112_unranked.md')

# ===== 模型加载 (与 t1 相同的预测口径) =====
with open(MODEL_PATH, 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']
LV_ORDER = m.get('lv_order', ['EZ', 'HD', 'IN', 'AT'])
CAPS = m.get('caps', {})
FLAT = m.get('MANUAL_FLAT', _MANUAL_FLAT_BOOT)

MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
MF3_GE30 = 0.50
MF3_LE5 = 1.0
MF3_MID = 0.8
EFF_GE30 = 1.0
EFF_LE5 = 1.50
EFF_MID = 1.0

DOMAIN_DELTA = {}
try:
    with open(ALIGN_PATH, encoding='utf-8') as _f:
        DOMAIN_DELTA = json.load(_f).get('delta', {})
except Exception:
    pass

def level_key(s):
    s = (s or '').upper()
    if 'AT' in s: return 'AT'
    if 'IN' in s: return 'IN'
    if 'HD' in s: return 'HD'
    return 'IN'

def level_onehot(lv):
    lv = level_key(lv)
    if 'IN_AT' in LV_ORDER and lv in ('IN', 'AT'):
        lv = 'IN_AT'
    if lv not in LV_ORDER:
        lv = LV_ORDER[-1]
    vec = [0.0] * len(LV_ORDER)
    vec[LV_ORDER.index(lv)] = 1.0
    return vec

def predict(feats_raw, level, do_calib=True):
    feats = dict(feats_raw)
    if level_key(level) == 'IN':
        for k, d in DOMAIN_DELTA.items():
            if k in feats:
                feats[k] = feats[k] - d
    x = np.array([[feats.get(n, 0) for n in FN] + level_onehot(level)])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    mf3 = feats_raw.get('multi_finger_3plus_events', 0)
    if mf3 >= 30:
        dens = feats_raw.get('above_avg_density_mean', 0)
        mf_scale = 0.70 if dens >= 12.5 else MF3_GE30
    else:
        mf_scale = MF3_LE5 if mf3 <= 5 else MF3_MID
    eff_scale = EFF_GE30 if mf3 >= 30 else (EFF_LE5 if mf3 <= 5 else EFF_MID)
    total = 0.0
    cd = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0)
        pv = P95.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t:
            continue
        e = v / t - 1.0
        c = CAPS.get(fname, cd)
        if c is not None and e > c:
            e = c
        co2 = co
        if fname in MF_FEATS:
            co2 = co * mf_scale
        elif fname in EFF_FEATS:
            co2 = co * eff_scale
        x_ = co2 * (e ** 0.70)
        p99 = max(P99.get(fname, 0), bl * 0.5)
        if v > p99:
            pe = v / p99 - 1.0
            if c is not None and pe > c:
                pe = c
            x_ += co2 * max(0, pe) ** 0.70 * 0.5
        total += x_
    pred = p_gb + total
    if do_calib:
        if 14 < pred <= 15:
            pred -= 0.30
        elif 15 < pred <= 16:
            pred -= 0.18
        elif 16 < pred <= 17:
            pred -= 0.05
    return pred

# ===== 官谱基准 (P90/P99, 16+段) =====
with open(CACHE_PATH, 'rb') as f:
    cache = pickle.load(f)
official = cache['official']
off_16 = [r for r in official if 16 <= r['diff'] < 17]
off_17 = [r for r in official if r['diff'] >= 17]
off_ge16 = [r for r in official if r['diff'] >= 16]

def q(vals, q):
    vals = sorted(vals)
    if not vals:
        return None
    idx = max(0, min(int(np.ceil(q * len(vals))) - 1, len(vals) - 1))
    return vals[idx]

# ===== 提取 + 预测 957 张 =====
lst = json.load(open(LIST_PATH, encoding='utf-8'))
meta = {c['id']: c for c in lst}

KEY_FEATS = ['above_avg_density_mean', 'real_core_notes_per_second',
             'eff_avg_tps_1s', 'eff_peak_tps_1s',
             'multi_finger_3plus_events', 'multi_finger_4plus_events',
             'weighted_mf_score_per_sec', 'multi_line_sim_events',
             'stair_density', 'stair_speed_avg', 'movement_per_second',
             'speed_volatility', 'tempo_change_count', 'total_notes', 'duration_sec']

t0 = time.time()
rows = []
fails = []
for c in lst:
    cid = c['id']
    path = os.path.join(JSON_DIR, f'{cid}.json')
    if not os.path.exists(path):
        fails.append((cid, 'missing'))
        continue
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        cd, _ = load_chart_from_bytes(raw)
        if cd is None:
            fails.append((cid, 'parse None'))
            continue
        feats = extract_features(cd, speed=1.0)
        if not feats:
            fails.append((cid, 'no feats'))
            continue
        lv = level_key(c.get('level', 'IN'))
        pred = predict(feats, lv)
        rows.append({'id': cid, 'name': c.get('name', ''), 'level': lv,
                     'diff': float(c.get('difficulty', np.nan)),
                     'pred': pred, 'feats': feats})
    except Exception as e:
        fails.append((cid, str(e)[:60]))
print(f'提取+预测完成: {len(rows)} 张, 失败 {len(fails)}, 耗时 {time.time()-t0:.1f}s')
for cid, err in fails[:10]:
    print(f'  FAIL {cid}: {err}')

lines = []
lines.append('=' * 78)
lines.append('v11.2 t3: 未上架957谱预测分布与官谱域对齐检查')
lines.append('预测口径: GB + 条件boost(mf3衰减/eff抬升) + 校准, 不含定轨加成')
lines.append(f'提取: json_unranked/*.json + unified_parser + extract_features(speed=1.0)')
lines.append(f'成功 {len(rows)} / 失败 {len(fails)}')
lines.append('=' * 78)

# ===== 分布统计 =====
def stat_block(vals, label):
    vals = np.array(vals)
    lines.append(f'{label}: n={len(vals)}')
    lines.append(f'  均值={vals.mean():.3f} 中位={np.median(vals):.3f} std={vals.std():.3f}')
    lines.append(f'  P10={np.percentile(vals,10):.3f} P25={np.percentile(vals,25):.3f} '
                 f'P75={np.percentile(vals,75):.3f} P90={np.percentile(vals,90):.3f} P99={np.percentile(vals,99):.3f}')
    lines.append(f'  min={vals.min():.3f} max={vals.max():.3f}')
    return vals

lines.append('')
lines.append('===== 分布统计 =====')
preds = stat_block([r['pred'] for r in rows], '未上架957 预测值')
diffs = stat_block([r['diff'] for r in rows if not np.isnan(r['diff'])], '未上架957 社区diff')
off_d = stat_block([r['diff'] for r in official], '官谱982 定数')
lines.append('')
lines.append('未上架预测值分段占比:')
seg_names = ['<13', '13-14', '14-15', '15-16', '16-17', '17-18', '18-19', '>=19']
for lo, hi, nm in [(0, 13, '<13'), (13, 14, '13-14'), (14, 15, '14-15'), (15, 16, '15-16'),
                   (16, 17, '16-17'), (17, 18, '17-18'), (18, 19, '18-19'), (19, 99, '>=19')]:
    n = sum(1 for r in rows if lo < r['pred'] <= hi) if hi < 99 else sum(1 for r in rows if r['pred'] >= lo)
    lines.append(f'  {nm}: {n} 张 ({n/len(rows)*100:.1f}%)')
gt18 = sum(1 for r in rows if r['pred'] > 18)
gt19 = sum(1 for r in rows if r['pred'] > 19)
lines.append(f'  >18: {gt18} 张 ({gt18/len(rows)*100:.1f}%) | >19: {gt19} 张 ({gt19/len(rows)*100:.1f}%)')

# 与社区 diff 偏差
lines.append('')
lines.append('===== 与社区 diff 偏差 (pred - diff) =====')
vd = [r for r in rows if not np.isnan(r['diff'])]
bias = np.array([r['pred'] - r['diff'] for r in vd])
lines.append(f'n={len(vd)} bias均值={bias.mean():+.3f} 中位={np.median(bias):+.3f} MAE={np.abs(bias).mean():.3f}')
lines.append(f'bias P10={np.percentile(bias,10):+.3f} P90={np.percentile(bias,90):+.3f}')
lines.append('偏差分段 (按社区 diff):')
for lo, hi, nm in [(16, 16.5, '16-16.5'), (16.5, 17, '16.5-17'), (17, 17.5, '17-17.5'), (17.5, 18, '17.5-18')]:
    sub = [r for r in vd if lo < r['diff'] <= hi]
    if sub:
        bb = np.mean([r['pred'] - r['diff'] for r in sub])
        lines.append(f'  {nm}: n={len(sub)} bias={bb:+.3f}')
# 偏差最大的10张
lines.append('')
lines.append('偏差最大 10 张 (高估):')
for r in sorted(vd, key=lambda r: -(r['pred'] - r['diff']))[:10]:
    lines.append(f'  id={r["id"]} {r["name"][:30]:<30} diff={r["diff"]:.2f} pred={r["pred"]:.2f} bias={r["pred"]-r["diff"]:+.2f}')
lines.append('偏差最小 10 张 (低估):')
for r in sorted(vd, key=lambda r: r['pred'] - r['diff'])[:10]:
    lines.append(f'  id={r["id"]} {r["name"][:30]:<30} diff={r["diff"]:.2f} pred={r["pred"]:.2f} bias={r["pred"]-r["diff"]:+.2f}')

# ===== 特征域检查 =====
lines.append('')
lines.append('=' * 78)
lines.append('===== 特征域检查: 未上架957 vs 官谱 16+ 段 (P90/P99) =====')
lines.append('=' * 78)
def feat_stats(rs, feat):
    vals = [r['feats'].get(feat, 0) for r in rs]
    return vals

lines.append(f'{"特征":<32}{"未上架P50":>10}{"未上架P90":>10}{"未上架P99":>10}{"官谱16-17P90":>14}{"官谱>=17P90":>12}')
for f in KEY_FEATS:
    ur = feat_stats(rows, f)
    p50 = np.percentile(ur, 50); p90 = np.percentile(ur, 90); p99 = np.percentile(ur, 99)
    o16 = q([r['feats'].get(f, 0) for r in off_16], 0.90)
    o17 = q([r['feats'].get(f, 0) for r in off_17], 0.90)
    lines.append(f'{f:<32}{p50:>10.2f}{p90:>10.2f}{p99:>10.2f}{o16:>14.2f}{o17:>12.2f}')

lines.append('')
lines.append('未上架谱特征超官谱 16-17 段 P90 的占比:')
lines.append(f'{"特征":<32}{"超官谱16-17P90":>14}')
for f in KEY_FEATS:
    o16_p90 = q([r['feats'].get(f, 0) for r in off_16], 0.90)
    if o16_p90 is None or o16_p90 <= 0:
        continue
    n = sum(1 for r in rows if r['feats'].get(f, 0) > o16_p90)
    lines.append(f'{f:<32}{n:>10} ({n/len(rows)*100:.1f}%)')

# mf3 分组
lines.append('')
lines.append('未上架谱 mf3 分组:')
mf3_groups = defaultdict(list)
for r in rows:
    mf3 = r['feats'].get('multi_finger_3plus_events', 0)
    g = '多指(mf3>=30)' if mf3 >= 30 else ('双指(mf3<=5)' if mf3 <= 5 else '混合')
    mf3_groups[g].append(r['pred'])
for g, ps in mf3_groups.items():
    lines.append(f'  {g}: n={len(ps)} pred均值={np.mean(ps):.2f} 中位={np.median(ps):.2f}')

with open(OUT_TXT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print(f'已输出: {OUT_TXT}')

# 供后续分析用的 JSON 摘要
import io
with open(os.path.join(_ROOT, 'logs', '_unranked_rows.json'), 'w', encoding='utf-8') as f:
    json.dump([{'id': r['id'], 'name': r['name'], 'level': r['level'], 'diff': r['diff'], 'pred': r['pred']} for r in rows], f, ensure_ascii=False)
print('已保存 _unranked_rows.json (供笔记分析)')
