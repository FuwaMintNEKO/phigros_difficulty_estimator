# -*- coding: utf-8 -*-
"""v11.2 实验 t1: 上架589谱偏差Top分析与特征模式定位

用 feats_cache_v11.pkl + 6dim_model_v11_1.pkl 预测上架谱
(GB + 条件boost mf3衰减/eff抬升 + 校准, 不含定轨加成), 计算 pred - diff 偏差:
  - 按段(13-14/14-15/15-16/16-17/17+)统计
  - 偏差最大/最小的各 20 张谱 (高估 Top20 / 低估 Top20)
  - 对 Top20 谱逐一列出关键特征值, 与官谱同段(按diff)的 P75/P90 对比
  - 总结高估/低估谱的共同特征模式

输出: logs/exp_v112_bias_top.txt + tools/notes_v112_bias_pattern.md
"""
import os, sys, json, pickle
import numpy as np
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT as _MANUAL_FLAT_BOOT

MODEL_PATH = os.path.join(_ROOT, 'models', '6dim_model_v11_1.pkl')
CACHE_PATH = os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl')
ALIGN_PATH = os.path.join(_ROOT, 'data', 'domain_align.json')
OUT_TXT = os.path.join(_ROOT, 'logs', 'exp_v112_bias_top.txt')
OUT_MD = os.path.join(_ROOT, 'tools', 'notes_v112_bias_pattern.md')

# ===== 模型加载 =====
with open(MODEL_PATH, 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']
LV_ORDER = m.get('lv_order', ['EZ', 'HD', 'IN', 'AT'])
CAPS = m.get('caps', {})
FLAT = m.get('MANUAL_FLAT', _MANUAL_FLAT_BOOT)

MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
MF3_GE30 = 0.50   # 低密度多指谱(堆料型) mf系数
MF3_LE5 = 1.0     # 双指谱不动
MF3_MID = 0.8     # 混合
EFF_GE30 = 1.0
EFF_LE5 = 1.50    # 双指谱 eff 抬升
EFF_MID = 1.0

# 自制谱 IN 段密度特征对齐 (与生产 app.py 一致)
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
    """v11.1 生产预测口径 (不含定轨加成): domain_align(IN) + GB + 条件boost + 校准"""
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

# ===== 数据加载 =====
with open(CACHE_PATH, 'rb') as f:
    cache = pickle.load(f)
official = cache['official']
ranked = cache['ranked']

def seg(d):
    if d < 13: return '<13'
    if d < 14: return '13-14'
    if d < 15: return '14-15'
    if d < 16: return '15-16'
    if d < 17: return '16-17'
    return '>=17'

SEGS = ['13-14', '14-15', '15-16', '16-17', '>=17']
valid = [r for r in ranked if r.get('diff') and r['diff'] > 10]
print(f'有效上架谱: {len(valid)}')

# ===== 预测 + 偏差 =====
results = []
for r in valid:
    pred = predict(r['feats'], r['level'])
    results.append({'id': r['id'], 'name': r['name'], 'level': r['level'],
                    'diff': r['diff'], 'pred': pred, 'bias': pred - r['diff'],
                    'feats': r['feats']})

# ===== 分段统计 =====
lines = []
lines.append('=' * 78)
lines.append('v11.2 t1: 上架589谱偏差Top分析 (feats_cache_v11.pkl + 6dim_model_v11_1.pkl)')
lines.append('预测口径: GB + 条件boost(mf3衰减/eff抬升) + 校准, 不含定轨加成')
lines.append('=' * 78)
lines.append('')
lines.append('===== 分段偏差统计 (按社区 diff 分段) =====')
bins = defaultdict(lambda: {'n': 0, 'b': 0.0, 'mae': 0.0, 'rmse': 0.0})
for r in results:
    b = bins[seg(r['diff'])]
    b['n'] += 1
    b['b'] += r['bias']
    b['mae'] += abs(r['bias'])
    b['rmse'] += r['bias'] ** 2
lines.append(f'{"段":<8}{"n":>5}{"bias均值":>10}{"MAE":>8}{"RMSE":>8}')
for k in SEGS + ['<13']:
    b = bins[k]
    if b['n'] == 0:
        continue
    rmse = (b['rmse'] / b['n']) ** 0.5
    lines.append(f'{k:<8}{b["n"]:>5}{b["b"] / b["n"]:>+10.3f}{b["mae"] / b["n"]:>8.3f}{rmse:>8.3f}')
all_b = sum(r['bias'] for r in results)
all_mae = sum(abs(r['bias']) for r in results)
all_rmse = sum(r['bias'] ** 2 for r in results)
lines.append(f'{"ALL":<8}{len(results):>5}{all_b / len(results):>+10.3f}{all_mae / len(results):>8.3f}{(all_rmse / len(results)) ** 0.5:>8.3f}')

# 16+ 分组 (多指/双指/混合)
hi = [r for r in results if r['diff'] >= 16]
groups = defaultdict(lambda: {'n': 0, 'b': 0.0})
for r in hi:
    mf3 = r['feats'].get('multi_finger_3plus_events', 0)
    g = '多指(mf3>=30)' if mf3 >= 30 else ('双指(mf3<=5)' if mf3 <= 5 else '混合')
    gr = groups[g]
    gr['n'] += 1
    gr['b'] += r['bias']
lines.append('')
lines.append('16+ 分组:')
for g, gr in groups.items():
    lines.append(f'  {g}: n={gr["n"]} bias={gr["b"] / gr["n"]:+.3f}')

# ===== Top20 =====
KEY_FEATS = ['above_avg_density_mean', 'real_core_notes_per_second',
             'eff_avg_tps_1s', 'eff_peak_tps_1s',
             'multi_finger_3plus_events', 'multi_finger_4plus_events',
             'weighted_mf_score_per_sec', 'multi_line_sim_events',
             'stair_density', 'stair_speed_avg', 'movement_per_second',
             'speed_volatility', 'tempo_change_count', 'total_notes', 'duration_sec']

def fmt_top(rs, title, key):
    lines.append('')
    lines.append('=' * 78)
    lines.append(f'===== {title} (按 |bias| {key}) =====')
    lines.append('=' * 78)
    lines.append(f'{"#":>3} {"id":>8} {"diff":>6} {"pred":>7} {"bias":>7}  name')
    for i, r in enumerate(rs, 1):
        nm = (r['name'] or '')[:34]
        lines.append(f'{i:>3} {r["id"]:>8} {r["diff"]:>6.2f} {r["pred"]:>7.2f} {r["bias"]:>+7.2f}  {nm}')
    return rs

over = sorted(results, key=lambda r: -r['bias'])[:20]   # 高估 Top20
under = sorted(results, key=lambda r: r['bias'])[:20]    # 低估 Top20
fmt_top(over, '高估 Top20 (bias 最大)', 'max')
fmt_top(under, '低估 Top20 (bias 最小)', 'min')

# ===== 官谱 P75/P90 (按 diff 段) =====
def quantile_vals(feat, rs, q):
    vals = sorted(r['feats'].get(feat, 0) for r in rs)
    if not vals:
        return None
    idx = int(np.ceil(q * len(vals))) - 1
    idx = max(0, min(idx, len(vals) - 1))
    return vals[idx]

official_by_seg = {}
for s in SEGS + ['<13']:
    rs = [r for r in official if seg(r['diff']) == s]
    official_by_seg[s] = rs

def feat_stats_table(rs, tag):
    """对一组谱输出特征值 + 官谱同段 P75/P90 对比表"""
    lines.append('')
    lines.append(f'--- {tag}: 关键特征 vs 官谱同段 P75/P90 ---')
    # 官谱同段 P75/P90 (按各谱所在段)
    over_p90 = Counter(); over_p75 = Counter()
    for f in KEY_FEATS:
        pass
    for r in rs:
        s = seg(r['diff'])
        off_rs = official_by_seg[s]
        for f in KEY_FEATS:
            v = r['feats'].get(f, 0)
            p75 = quantile_vals(f, off_rs, 0.75)
            p90 = quantile_vals(f, off_rs, 0.90)
            if p90 is not None and v > p90:
                over_p90[f] += 1
            elif p75 is not None and v > p75:
                over_p75[f] += 1
    n = len(rs)
    lines.append(f'特征超官谱同段范围统计 (n={n}):')
    lines.append(f'{"特征":<32}{"超P90数":>8}{"超P75数":>8}')
    for f in KEY_FEATS:
        if over_p90[f] > 0 or over_p75[f] > 0:
            lines.append(f'{f:<32}{over_p90[f]:>8}{over_p75[f]:>8}')
    lines.append('')
    # 逐谱明细
    hdr = f'{"#":>3} {"diff":>6} {"bias":>7}  ' + ' '.join(f'{f[:10]:>11}' for f in KEY_FEATS[:7])
    lines.append(hdr)
    lines.append('-' * len(hdr))
    for i, r in enumerate(rs, 1):
        row = f'{i:>3} {r["diff"]:>6.2f} {r["bias"]:>+7.2f}  '
        for f in KEY_FEATS[:7]:
            v = r['feats'].get(f, 0)
            row += f'{v:>11.2f}'
        lines.append(row)
    lines.append('')
    hdr2 = f'{"#":>3} {"diff":>6} {"bias":>7}  ' + ' '.join(f'{f[:10]:>11}' for f in KEY_FEATS[7:])
    lines.append(hdr2)
    lines.append('-' * len(hdr2))
    for i, r in enumerate(rs, 1):
        row = f'{i:>3} {r["diff"]:>6.2f} {r["bias"]:>+7.2f}  '
        for f in KEY_FEATS[7:]:
            v = r['feats'].get(f, 0)
            row += f'{v:>11.2f}'
        lines.append(row)
    return over_p90, over_p75

lines.append('')
lines.append('=' * 78)
lines.append('官谱同段 P75/P90 (作为对比基准)')
lines.append('=' * 78)
for s in SEGS:
    off_rs = official_by_seg[s]
    lines.append(f'段 {s}: 官谱 n={len(off_rs)}')
    for f in KEY_FEATS[:7]:
        p75 = quantile_vals(f, off_rs, 0.75)
        p90 = quantile_vals(f, off_rs, 0.90)
        lines.append(f'  {f:<32} P75={p75:.3f} P90={p90:.3f}')
    lines.append('')

# ===== 增强: 完整官谱 P75/P90 表 =====
lines.append('')
lines.append('=' * 78)
lines.append('官谱同段 P75/P90 完整表 (全部关键特征)')
lines.append('=' * 78)
lines.append(f'{"特征":<32}' + ''.join(f'{s:>16}' for s in SEGS))
for f in KEY_FEATS:
    row = f'{f:<32}'
    for s in SEGS:
        off_rs = official_by_seg[s]
        p75 = quantile_vals(f, off_rs, 0.75)
        p90 = quantile_vals(f, off_rs, 0.90)
        row += f'{p75:.2f}/{p90:.2f}'.rjust(16)
    lines.append(row)
lines.append('  格式: P75/P90; >=17 段官谱仅13张, P值仅供参考')

# ===== 增强: 方向性对比统计 =====
def feat_dir_stats(rs, tag, over=True):
    lines.append('')
    lines.append(f'--- {tag}: 特征相对官谱同段范围的方向性统计 (n={len(rs)}) ---')
    up_p90 = Counter(); up_p75 = Counter(); dn_p25 = Counter(); dn_p10 = Counter()
    for r in rs:
        s = seg(r['diff'])
        off_rs = official_by_seg[s]
        for f in KEY_FEATS:
            v = r['feats'].get(f, 0)
            p90 = quantile_vals(f, off_rs, 0.90)
            p75 = quantile_vals(f, off_rs, 0.75)
            p25 = quantile_vals(f, off_rs, 0.25)
            p10 = quantile_vals(f, off_rs, 0.10)
            if p90 is not None and v > p90: up_p90[f] += 1
            elif p75 is not None and v > p75: up_p75[f] += 1
            if p25 is not None and v < p25: dn_p25[f] += 1
            if p10 is not None and v < p10: dn_p10[f] += 1
    lines.append(f'{"特征":<32}{"超P90":>7}{"超P75":>7}{"低P25":>7}{"低P10":>7}')
    for f in KEY_FEATS:
        lines.append(f'{f:<32}{up_p90[f]:>7}{up_p75[f]:>7}{dn_p25[f]:>7}{dn_p10[f]:>7}')
    return up_p90, up_p75, dn_p25, dn_p10

ov_p90, ov_p75, _, _ = feat_dir_stats(over, '高估 Top20')
un_p90, un_p75, un_dn25, un_dn10 = feat_dir_stats(under, '低估 Top20')

# ===== 增强: 逐谱特征值 vs 官谱同段 P75/P90 (带标记) =====
def per_chart_feats(rs, tag):
    lines.append('')
    lines.append('=' * 78)
    lines.append(f'{tag}: 逐谱特征值 vs 官谱同段 P75/P90 (▲=超P90 △=超P75 ▼=低P25 ▽=低P10)')
    lines.append('=' * 78)
    for i, r in enumerate(rs, 1):
        s = seg(r['diff'])
        off_rs = official_by_seg[s]
        nm = (r['name'] or '')[:30]
        lines.append(f'[{i}] {nm} (id={r["id"]}, diff={r["diff"]:.2f}, pred={r["pred"]:.2f}, bias={r["bias"]:+.2f}, 段={s})')
        for f in KEY_FEATS:
            v = r['feats'].get(f, 0)
            p75 = quantile_vals(f, off_rs, 0.75)
            p90 = quantile_vals(f, off_rs, 0.90)
            p25 = quantile_vals(f, off_rs, 0.25)
            p10 = quantile_vals(f, off_rs, 0.10)
            mark = ''
            if p90 is not None and v > p90: mark = ' ▲'
            elif p75 is not None and v > p75: mark = ' △'
            elif p25 is not None and v < p25: mark = ' ▼'
            elif p10 is not None and v < p10: mark = ' ▽'
            lines.append(f'    {f:<32} {v:>12.2f}  (官谱P75={p75:.2f} P90={p90:.2f} P25={p25:.2f} P10={p10:.2f}){mark}')
    return None

per_chart_feats(over, '高估 Top20 逐谱明细')
per_chart_feats(under, '低估 Top20 逐谱明细')

# ===== 增强: 全体589 vs Top20 特征中位数对比 =====
def med(vals):
    vals = sorted(vals)
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2

lines.append('')
lines.append('=' * 78)
lines.append('特征中位数对比: 全体589 vs 高估Top20 vs 低估Top20 vs 官谱>=14段')
lines.append('=' * 78)
off_ge14 = [r for r in official if r['diff'] >= 14]
lines.append(f'{"特征":<32}{"589":>10}{"高估20":>10}{"低估20":>10}{"官谱>=14":>10}')
for f in KEY_FEATS:
    a = med([r['feats'].get(f, 0) for r in results])
    b = med([r['feats'].get(f, 0) for r in over])
    c = med([r['feats'].get(f, 0) for r in under])
    d = med([r['feats'].get(f, 0) for r in off_ge14])
    lines.append(f'{f:<32}{a:>10.2f}{b:>10.2f}{c:>10.2f}{d:>10.2f}')

with open(OUT_TXT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print(f'已输出: {OUT_TXT}')

# ===== 保存 txt =====
with open(OUT_TXT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print(f'已输出: {OUT_TXT}')
print('\n'.join(lines[:40]))