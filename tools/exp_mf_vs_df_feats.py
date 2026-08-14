# -*- coding: utf-8 -*-
"""t2: 16+ 段多指谱 vs 双指谱特征分离度实验
分组: multi_finger_3plus_events >= 30 多指组 / <= 5 双指组
分离度 = |均值差| / 合并标准差(pooled std)
输出: logs/exp_mf_vs_df_feats.txt
"""
import os, sys, json, csv, io
import numpy as np

_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from feature_extractor import extract_features
from unified_parser import load_chart_from_bytes

OUT = os.path.join(_ROOT, 'logs', 'exp_mf_vs_df_feats.txt')
JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json')
PRED_CSV = os.path.join(_ROOT, 'data', 'phira', 'predictions.csv')
CHART_META = os.path.join(_ROOT, 'data', 'phira', 'charts.json')

TARGET_FEATS = [
    'weighted_mf_score_per_sec',
    'eff_peak_tps_1s',
    'eff_avg_tps_1s',
    'stair_speed_avg',
    'chord_alternation_rate',
    'above_avg_density_mean',
    'above_avg_duration_sec',
    'real_core_notes_per_second',
    'movement_per_second',
    'jline_movement_density',
    'type_switch_per_sec',
    'tap_burst_top5',
    'fast_note_density_32nd',
]
MF3 = 'multi_finger_3plus_events'
THRESH_HI = 30   # >=30 多指组
THRESH_LO = 5    # <=5 双指组

def _buf(*args):
    line = ' '.join(str(a) for a in args)
    print(line)
    return line + '\n'

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3: return float('nan')
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.dot(rx, ry) / len(rx))

def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12: return float('nan')
    return float(np.corrcoef(x, y)[0, 1])

def main():
    # 1) 社区定数
    diff_by_id = {}
    with open(PRED_CSV, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            try:
                diff_by_id[int(row['id'])] = float(row['diff'])
            except Exception:
                pass
    charts = json.load(open(CHART_META, encoding='utf-8'))
    meta_by_id = {}
    for lst in charts.values():
        for c in lst:
            meta_by_id[c['id']] = c

    # 2) 提取 16+ 特征
    rows = []
    skipped = []
    for fn in sorted(os.listdir(JSON_DIR)):
        if not fn.endswith('.json'): continue
        cid = int(fn[:-5])
        diff = diff_by_id.get(cid)
        if diff is None or diff < 16.0:
            continue
        try:
            with open(os.path.join(JSON_DIR, fn), 'rb') as f:
                chart_data, _ = load_chart_from_bytes(f.read())
            feats = extract_features(chart_data)
            if not feats:
                skipped.append((cid, 'extract None')); continue
            meta = meta_by_id.get(cid, {})
            rows.append({
                'id': cid, 'name': meta.get('name', ''), 'level': meta.get('level', ''),
                'diff': diff, 'feats': feats,
            })
        except Exception as e:
            skipped.append((cid, str(e)[:80]))
    print(f'16+ 谱面提取成功: {len(rows)} (跳过 {len(skipped)})')
    if skipped:
        for cid, why in skipped[:10]:
            print(f'  skip {cid}: {why}')

    # 3) 分组
    mf3 = np.array([r['feats'].get(MF3, 0) for r in rows])
    hi = [r for r, v in zip(rows, mf3) if v >= THRESH_HI]
    lo = [r for r, v in zip(rows, mf3) if v <= THRESH_LO]
    mid = [r for r, v in zip(rows, mf3) if THRESH_LO < v < THRESH_HI]
    print(f'多指组(mf3>={THRESH_HI}): {len(hi)}  双指组(mf3<={THRESH_LO}): {len(lo)}  中间带: {len(mid)}')

    out = []
    out.append(_buf('=' * 100))
    out.append(_buf('t2: 16+ 段多指谱 vs 双指谱特征分离度实验'))
    out.append(_buf(f'样本: 16+ 段 {len(rows)} 张 | 多指组(mf3>={THRESH_HI}) {len(hi)} | 双指组(mf3<={THRESH_LO}) {len(lo)} | 中间带 {len(mid)}'))

    # 4) 分离度对比
    out.append(_buf(''))
    out.append(_buf('--- 特征分离度: |均值差|/合并std (pooled) ---'))
    out.append(_buf(f'{"feature":<30} {"hi均值":>10} {"lo均值":>10} {"hi std":>9} {"lo std":>9} {"分离度":>7} {"方向":>6}'))
    sep_rows = []
    for fname in TARGET_FEATS + ['multi_finger_3plus_ratio', 'multi_finger_density', 'notes_per_second']:
        hv = np.array([r['feats'].get(fname, 0) for r in hi], float)
        lv = np.array([r['feats'].get(fname, 0) for r in lo], float)
        if len(hv) < 2 or len(lv) < 2:
            continue
        hm, lm = hv.mean(), lv.mean()
        hs, ls = hv.std(ddof=1), lv.std(ddof=1)
        pooled = np.sqrt(((len(hv)-1)*hs*hs + (len(lv)-1)*ls*ls) / (len(hv)+len(lv)-2))
        sep = abs(hm - lm) / pooled if pooled > 1e-12 else float('inf')
        direction = '多指>' if hm > lm else '多指<'
        out.append(_buf(f'{fname:<30} {hm:>10.4f} {lm:>10.4f} {hs:>9.4f} {ls:>9.4f} {sep:>7.3f} {direction:>6}'))
        sep_rows.append((fname, sep, hm, lm))

    # 5) 与社区定数相关性 (16+ 全体 + 两组)
    out.append(_buf(''))
    out.append(_buf('--- 特征与社区定数(diff)相关性 ---'))
    out.append(_buf(f'{"feature":<30} {"全体Pearson":>12} {"全体Spearman":>13} {"多指组r":>9} {"双指组r":>9}'))
    corr_rows = []
    all_diff = np.array([r['diff'] for r in rows], float)
    for fname in TARGET_FEATS:
        av = np.array([r['feats'].get(fname, 0) for r in rows], float)
        hv = np.array([r['feats'].get(fname, 0) for r in hi], float)
        lv = np.array([r['feats'].get(fname, 0) for r in lo], float)
        hd = np.array([r['diff'] for r in hi], float)
        ld = np.array([r['diff'] for r in lo], float)
        pr = pearson(av, all_diff); sr = spearman(av, all_diff)
        hr = pearson(hv, hd); lr = pearson(lv, ld)
        out.append(_buf(f'{fname:<30} {pr:>12.4f} {sr:>13.4f} {hr:>9.4f} {lr:>9.4f}'))
        corr_rows.append((fname, pr, sr))

    # 6) 排序总结: 分离度 Top + 与定数相关 Top
    out.append(_buf(''))
    out.append(_buf('--- 分离度排序 (最能区分多指/双指的特征) ---'))
    for i, (fname, sep, hm, lm) in enumerate(sorted(sep_rows, key=lambda t: -t[1]), 1):
        out.append(_buf(f'{i:>2}. {fname:<30} sep={sep:.3f} (多指均值 {hm:.3f} vs 双指均值 {lm:.3f})'))
    out.append(_buf(''))
    out.append(_buf('--- 与社区定数 Spearman 相关排序 (16+ 全体) ---'))
    for i, (fname, pr, sr) in enumerate(sorted(corr_rows, key=lambda t: -abs(t[2])), 1):
        out.append(_buf(f'{i:>2}. {fname:<30} pearson={pr:+.3f} spearman={sr:+.3f}'))

    # 7) 两组基线统计
    out.append(_buf(''))
    out.append(_buf('--- 两组基线 ---'))
    for tag, grp in (('多指组', hi), ('双指组', lo)):
        if not grp: continue
        ds = np.array([r['diff'] for r in grp], float)
        ps = np.array([r['feats'].get('notes_per_second', 0) for r in grp], float)
        m3 = np.array([r['feats'].get(MF3, 0) for r in grp], float)
        out.append(_buf(f'{tag}: n={len(grp)} 社区定数 {ds.mean():.2f}±{ds.std(ddof=1):.2f} ({ds.min():.1f}~{ds.max():.1f}) '
                        f'nps {ps.mean():.2f} mf3 {m3.mean():.0f}±{m3.std(ddof=1):.0f}'))
        names = ', '.join(f"{r['name']}({r['diff']:.1f})" for r in grp)
        out.append(_buf(f'   成员: {names}'))

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(''.join(out))
    print(f'\n结果已写入: {OUT}')

if __name__ == '__main__':
    main()
