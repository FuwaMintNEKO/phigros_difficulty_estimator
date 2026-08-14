# -*- coding: utf-8 -*-
"""boost 权重修正实验: 针对 Sigma(演出虚高) / Retribution(耐力偏低)
对若干权重变体, 用现有 v10 GB(不重训) 快速看趋势:
  - 官方谱全量拟合 MAE (只做变体间相对比较, 非诚实CV)
  - Sigma / Retribution 的 AT 预测
  - 有定数自制谱的外推/内推 MAE
确认组合后再用 train_final_v10.py 重训做诚实 CV 验证。
"""
import os, sys, re, pickle, copy, numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import app  # 载入 v10 模型 (gb/scaler/FN/P95/P99)
from feature_extractor import extract_features
from boost_config import MANUAL_FLAT
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from unified_parser import load_chart_from_bytes

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
DL = r'C:\Users\NaNK\Downloads'

# ---------- 候选权重变体 ----------
# 新特征 (音符级差速/jack): V0-V12 不使用, V13+ 启用
NEW_FEATS = ['jack_density', 'jack_max_run', 'same_line_jack_ratio', 'long_jack_count',
             'note_speed_non1_ratio', 'note_speed_std', 'note_speed_max',
             'note_speed_density', 'flash_hold_ratio', 'chord_jack_density',
             'chord_jack_3plus_pairs']
# 差速+和弦重键 (V14/V15 保留的核心新特征)
CORE_NEW = ('note_speed_non1_ratio', 'note_speed_std', 'note_speed_max', 'note_speed_density',
            'chord_jack_density', 'chord_jack_3plus_pairs')

def variant(name, changes, caps=None, remove=None):
    flat = [list(t) for t in MANUAL_FLAT]
    d = {f: (bl, co) for f, bl, co in flat}
    for f, new_co in changes.items():
        d[f] = (d[f][0], new_co)
    for f in (remove or []):
        d.pop(f, None)
    return name, [(f, bl, co) for f, (bl, co) in d.items()], caps or {}

VARIANTS = [
    variant('V0 基线', {}, remove=NEW_FEATS),
    variant('V1 温和: tsw/2 chord_alt-22% mf+25% end+10%',
            {'type_switch_per_sec': 0.05, 'chord_alternation_rate': 0.15,
             'weighted_mf_score_per_sec': 0.22, 'above_avg_duration_sec': 0.44},
            remove=NEW_FEATS),
    variant('V2 激进: tsw-70% chord_alt-38% mf+42% end+20%',
            {'type_switch_per_sec': 0.03, 'chord_alternation_rate': 0.12,
             'weighted_mf_score_per_sec': 0.25, 'above_avg_duration_sec': 0.48},
            remove=NEW_FEATS),
    variant('V3 只压type_switch: 0.04',
            {'type_switch_per_sec': 0.04}, remove=NEW_FEATS),
    variant('V4 只升多指耐力: mf+42% end+20%',
            {'weighted_mf_score_per_sec': 0.25, 'above_avg_duration_sec': 0.48},
            remove=NEW_FEATS),
    # ---- cap 方案 (对极端 excess 封顶, 正常谱不受影响) ----
    variant('V5 全局cap=4', {}, caps={'_default': 4.0}, remove=NEW_FEATS),
    variant('V6 只type_switch cap=4', {}, caps={'type_switch_per_sec': 4.0}, remove=NEW_FEATS),
    variant('V7 tsw cap3.5+co0.08  alt3.5 mf0.22 end0.44',
            {'type_switch_per_sec': 0.08, 'weighted_mf_score_per_sec': 0.22,
             'above_avg_duration_sec': 0.44},
            caps={'type_switch_per_sec': 3.5, 'chord_alternation_rate': 3.5}, remove=NEW_FEATS),
    variant('V8 tsw cap3+co0.06  alt3 mf0.24 end0.46',
            {'type_switch_per_sec': 0.06, 'weighted_mf_score_per_sec': 0.24,
             'above_avg_duration_sec': 0.46},
            caps={'type_switch_per_sec': 3.0, 'chord_alternation_rate': 3.0}, remove=NEW_FEATS),
    # ---- cap4 全局 + 温和升多指耐力 (平衡 Sigma 虚高 / Retri 低估) ----
    variant('V9  cap4 + mf0.20 end0.42', {'weighted_mf_score_per_sec': 0.20,
             'above_avg_duration_sec': 0.42}, caps={'_default': 4.0}, remove=NEW_FEATS),
    variant('V10 cap4 + mf0.22 end0.44', {'weighted_mf_score_per_sec': 0.22,
             'above_avg_duration_sec': 0.44}, caps={'_default': 4.0}, remove=NEW_FEATS),
    variant('V11 cap4 + tsw0.06 alt0.15 mf0.20 end0.42',
            {'type_switch_per_sec': 0.06, 'chord_alternation_rate': 0.15,
             'weighted_mf_score_per_sec': 0.20, 'above_avg_duration_sec': 0.42},
            caps={'_default': 4.0}, remove=NEW_FEATS),
    variant('V12 cap4 + tsw0.05 alt0.13 mf0.22 end0.46',
            {'type_switch_per_sec': 0.05, 'chord_alternation_rate': 0.13,
             'weighted_mf_score_per_sec': 0.22, 'above_avg_duration_sec': 0.46},
            caps={'_default': 4.0}, remove=NEW_FEATS),
    # ---- 新增特征 (差速/jack) + cap4 ----
    variant('V13 新特征全套 cap4', {}, caps={'_default': 4.0}),
    variant('V14 新特征仅差速+和弦重键 cap4', {}, remove=[f for f in NEW_FEATS if f not in CORE_NEW],
            caps={'_default': 4.0}),
    variant('V15 差速cap4 + tsw0.06 alt0.15 mf0.20 end0.42',
            {'type_switch_per_sec': 0.06, 'chord_alternation_rate': 0.15,
             'weighted_mf_score_per_sec': 0.20, 'above_avg_duration_sec': 0.42},
            remove=[f for f in NEW_FEATS if f not in CORE_NEW],
            caps={'_default': 4.0}),
]

def compute_boost(feats, flat, p95, p99, caps, speed=1.0):
    total = 0.0
    cap = caps.get('_default', None)
    for fname, bl, co in flat:
        v = feats.get(fname, 0)
        pv = p95.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t:
            continue
        e = v / t - 1.0
        c = caps.get(fname, cap)
        if c is not None and e > c:
            e = c
        x = co * (e ** 0.70)
        if v > max(p99.get(fname, 0), bl * 0.5):
            pe = v / max(p99.get(fname, 0), bl * 0.5) - 1.0
            if c is not None and pe > c:
                pe = c
            x += co * max(0, pe) ** 0.70 * 0.5
        total += x
    return total

def predict_with(feats, level, flat, caps, p95=None, p99=None):
    x = np.array([[feats.get(n, 0) for n in app.FN] + app._level_onehot(level)])
    xs = app.scaler.transform(x)
    g = float(app.gb.predict(xs)[0])
    b = compute_boost(feats, flat, p95 or app.P95, p99 or app.P99, caps)
    return g + b, g, b

# ---------- 官方谱 ----------
song_diffs = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)
official = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_diffs:
        continue
    diffs = song_diffs[sid]
    for lv in ['EZ', 'HD', 'IN', 'AT']:
        if lv in info['levels'] and lv in diffs:
            official.append({'path': info['levels'][lv], 'y': diffs[lv], 'lv': lv})

print(f'官方谱 {len(official)} 张, 提取特征...')
official_feats = []
for it in official:
    try:
        cd = load_chart_json(it['path'])
        f = extract_features(cd)
        if f:
            official_feats.append((f, it['y'], it['lv']))
    except Exception:
        pass
print(f'官方特征提取成功 {len(official_feats)}')

# 用官方谱重算全特征 P95/P99 (含新增差速/jack特征, 使新特征有合理阈值)
FLAT_KEYS = sorted({f for f, _, _ in MANUAL_FLAT})
p95_all = {k: float(np.percentile([f.get(k, 0) for f, _, _ in official_feats], 95)) for k in FLAT_KEYS}
p99_all = {k: float(np.percentile([f.get(k, 0) for f, _, _ in official_feats], 99)) for k in FLAT_KEYS}

def official_mae(flat, caps):
    errs = []
    for f, y, lv in official_feats:
        p, _, _ = predict_with(f, lv, flat, caps, p95_all, p99_all)
        errs.append(abs(p - y))
    return sum(errs) / len(errs)

# ---------- 自制谱 ----------
PAT = re.compile(r'^(.*?)\((\d+(?:\.\d+)?)(?:~(\d+(?:\.\d+)?))?\)(?:\(\d+\))?[^.]*\.json$')

def level_for(d):
    if d is None: return 'AT'
    if d >= 16.5: return 'AT'
    if d >= 11.5: return 'IN'
    if d >= 6.5: return 'HD'
    return 'EZ'

custom = []  # (name, user_d or None, feats)
for fn in sorted(os.listdir(DL)):
    if not fn.lower().endswith('.json'):
        continue
    m = PAT.match(fn)
    if m:
        name, a, b = m.group(1), float(m.group(2)), m.group(3)
        user_d = (a + float(b)) / 2 if b else a
    else:
        name, user_d = fn[:-5], None
    path = os.path.join(DL, fn)
    try:
        with open(path, 'rb') as fh:
            raw = fh.read()
        cd, _ = load_chart_from_bytes(raw)
        if cd is None:
            continue
        fe = extract_features(cd)
        if fe:
            custom.append((name, user_d, fe))
    except Exception:
        pass
print(f'自制谱 {len(custom)} 张 (含无定数), 特征提取成功')

# ---------- 结果 ----------
print('\n' + '=' * 100)
print(f'{"变体":<44} {"官方MAE":>8} {"SigmaAT":>8} {"RetriAT":>8} {"外推MAE":>8} {"内推MAE":>8}')
# 已知异常谱 (不计入内推统计): Chart_SP #1347 (定数1.0是bug) + スタートリップ (gimmick)
EXCLUDE_INTR = ('Chart_SP', 'スタートリップ')
for name, flat, caps in VARIANTS:
    omae = official_mae(flat, caps)
    sigma = retri = ext_mae = intr_mae = float('nan')
    for cname, ud, fe in custom:
        lv = level_for(ud)
        p, _, _ = predict_with(fe, lv, flat, caps, p95_all, p99_all)
        if cname.startswith('Sigma') and 'Regrets' in cname:
            sigma = p
        elif 'Retribution_FULL' in cname:
            retri = p
    ext_errs = []
    intr_errs = []
    for cname, ud, fe in custom:
        if ud is None:
            continue
        p, _, _ = predict_with(fe, level_for(ud), flat, caps, p95_all, p99_all)
        e = abs(p - ud)
        if ud > 17.5:
            ext_errs.append(e)
        elif not any(x in cname for x in EXCLUDE_INTR):
            intr_errs.append(e)
    ext_mae = sum(ext_errs) / len(ext_errs) if ext_errs else float('nan')
    intr_mae = sum(intr_errs) / len(intr_errs) if intr_errs else float('nan')
    print(f'{name:<44} {omae:>8.4f} {sigma:>8.2f} {retri:>8.2f} {ext_mae:>8.3f} {intr_mae:>8.3f}')

# 明细: 有定数自制谱在 V0 与 V14(新特征) 下的预测对比
print('\n' + '=' * 100)
flat0 = [t for n, t, c in VARIANTS if n.startswith('V0')][0]
caps0 = [c for n, t, c in VARIANTS if n.startswith('V0')][0]
flat2 = [t for n, t, c in VARIANTS if n.startswith('V14')][0]
caps2 = [c for n, t, c in VARIANTS if n.startswith('V14')][0]
print(f'{"谱面":<30} {"定数":>6} {"V0":>7} {"V14":>7} {"偏差":>7}')
lines_out = [f'{"谱面":<30} {"定数":>6} {"V0":>7} {"V14":>7} {"偏差":>7}']
for cname, ud, fe in sorted(custom, key=lambda t: -(t[1] or 0)):
    lv = level_for(ud)
    p0, _, _ = predict_with(fe, lv, flat0, caps0, p95_all, p99_all)
    p2, _, _ = predict_with(fe, lv, flat2, caps2, p95_all, p99_all)
    if ud is None:
        row = f'{str(cname)[:30]:<30} {"无":>6} {p0:>7.2f} {p2:>7.2f}'
        print(row)
    else:
        row = f'{str(cname)[:30]:<30} {ud:>6.1f} {p0:>7.2f} {p2:>7.2f} {p2-ud:>+7.2f}'
        print(row)
    lines_out.append(row)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_exp_detail.txt'), 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines_out))
