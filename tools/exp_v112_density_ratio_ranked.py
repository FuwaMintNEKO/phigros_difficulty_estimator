# -*- coding: utf-8 -*-
"""t2 补充验证: 上架谱 vs 官谱 eff_density_ratio 分布对比 + 谱例查找"""
import os, sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
_ROOT = r'D:\\Trae项目\\新建文件夹\\phigros_difficulty_estimator'
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)

def row_of(o, src):
    f = o['feats']
    dens = f.get('above_avg_density_mean', 0)
    effa = f.get('eff_avg_tps_1s', 0)
    rcnps = f.get('real_core_notes_per_second', 0)
    ratio = effa / max(dens, 0.1)
    redun = effa / max(rcnps, 0.1)
    return {'src': src, 'name': o.get('name', o.get('id','')), 'level': o.get('level',''),
            'diff': float(o['diff']) if o.get('diff') else None, 'dens': dens, 'effa': effa,
            'rcnps': rcnps, 'ratio': ratio, 'redun': redun,
            'mf3': f.get('multi_finger_3plus_events', 0), 'mf4': f.get('multi_finger_4plus_events', 0),
            'wmf': f.get('weighted_mf_score_per_sec', 0), 'nps': f.get('notes_per_second', 0)}

official_rows = [row_of(o, 'official') for o in cache['official']]
ranked_rows = [row_of(r, 'ranked') for r in cache['ranked'] if r.get('diff') and r['diff'] > 10]
print(f'官谱 {len(official_rows)} 上架(有效) {len(ranked_rows)}')

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3: return float('nan')
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12); ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.dot(rx, ry) / len(rx))

print()
print('===== 按定数段: 官谱 vs 上架谱 的 ratio/dens 分布 =====')
bins = [('13-14', 13, 14), ('14-15', 14, 15), ('15-16', 15, 16), ('16-17', 16, 17), ('>=17', 17, 99)]
print(f'{"段":<6}{"组":<10}{"n":>4}{"dens均值":>9}{"ratio中位":>11}{"ratio P25":>11}{"ratio P75":>11}{"effa均值":>9}{"wmf均值":>8}')
for name, lo, hi in bins:
    for src, grp in [('official', official_rows), ('ranked', ranked_rows)]:
        sel = [r for r in grp if lo <= r['diff'] < hi]
        if not sel: continue
        rr = np.array([r['ratio'] for r in sel])
        dd = np.array([r['dens'] for r in sel])
        ee = np.array([r['effa'] for r in sel])
        ww = np.array([r['wmf'] for r in sel])
        print(f'{name:<6}{src:<10}{len(sel):>4}{dd.mean():>9.2f}{np.median(rr):>11.3f}{np.percentile(rr,25):>11.3f}{np.percentile(rr,75):>11.3f}{ee.mean():>9.2f}{ww.mean():>8.2f}')

print()
print('===== 上架谱 ratio 最低的 15 张 (多押撑密度嫌疑最大) =====')
for r in sorted(ranked_rows, key=lambda x: x['ratio'])[:15]:
    print(f'  id={str(r["name"])[:12]:<14} diff={r["diff"]:>5.1f} dens={r["dens"]:>5.1f} effa={r["effa"]:>5.2f} ratio={r["ratio"]:>5.2f} wmf={r["wmf"]:>5.2f} mf3={r["mf3"]:>5.0f} nps={r["nps"]:>5.1f}')

print()
print('===== 查找: ギザバ怪文書 / Sigma Regrets =====')
for kw in ['ギザ', '怪文', 'Sigma', 'Regret']:
    for r in official_rows + ranked_rows:
        if kw.lower() in r['name'].lower():
            print(f'  [{r["src"]}] {r["name"][:50]} diff={r["diff"]} dens={r["dens"]:.2f} effa={r["effa"]:.2f} ratio={r["ratio"]:.3f} nps={r["nps"]:.1f}')

print()
print('===== 上架谱 ratio vs 官谱同段 (超官谱P90占比) =====')
for name, lo, hi in bins:
    off = [r['ratio'] for r in official_rows if lo <= r['diff'] < hi]
    rk = [r['ratio'] for r in ranked_rows if lo <= r['diff'] < hi]
    if not off or not rk: continue
    p90 = np.percentile(off, 90)
    over = np.mean(np.array(rk) > p90) * 100
    print(f'  {name:<6} 官谱ratio P90={p90:.3f}  上架ratio超P90占比={over:.1f}% (n={len(rk)})')
