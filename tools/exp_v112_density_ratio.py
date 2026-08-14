# -*- coding: utf-8 -*-
"""t2 方案A验证: eff_density_ratio 与现有密度特征对比 (官谱982)"""
import os, sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
_ROOT = r'D:\\Trae项目\\新建文件夹\\phigros_difficulty_estimator'
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = cache['official']
print(f'官谱数: {len(official)}')

def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12: return float('nan')
    return float(np.corrcoef(x, y)[0, 1])

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3: return float('nan')
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12); ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.dot(rx, ry) / len(rx))

rows = []
for o in official:
    f = o['feats']
    diff = float(o['diff'])
    dens = f.get('above_avg_density_mean', 0)
    effa = f.get('eff_avg_tps_1s', 0)
    effp = f.get('eff_peak_tps_1s', 0)
    rcnps = f.get('real_core_notes_per_second', 0)
    mf3 = f.get('multi_finger_3plus_events', 0)
    mf4 = f.get('multi_finger_4plus_events', 0)
    wmf = f.get('weighted_mf_score_per_sec', 0)
    ratio = effa / max(dens, 0.1)
    # 全窗冗余因子 (方案B近似的核心)
    redun = effa / max(rcnps, 0.1)
    rows.append({'name': o['name'], 'level': o['level'], 'diff': diff, 'dens': dens,
                 'effa': effa, 'effp': effp, 'rcnps': rcnps, 'ratio': ratio,
                 'redun': redun, 'mf3': mf3, 'mf4': mf4, 'wmf': wmf})

D = np.array([r['diff'] for r in rows])
feats = {
    'above_avg_density_mean': [r['dens'] for r in rows],
    'eff_avg_tps_1s': [r['effa'] for r in rows],
    'eff_peak_tps_1s': [r['effp'] for r in rows],
    'real_core_notes_per_second': [r['rcnps'] for r in rows],
    'eff_density_ratio (A)': [r['ratio'] for r in rows],
    'eff_avg/rcnps 冗余因子': [r['redun'] for r in rows],
    'multi_finger_3plus_events': [r['mf3'] for r in rows],
    'weighted_mf_score_per_sec': [r['wmf'] for r in rows],
}
print()
print('===== 各特征与官谱定数 diff 的相关性 =====')
print(f'{"特征":<32}{"Pearson":>10}{"Spearman":>10}')
for k, v in feats.items():
    print(f'{k:<32}{pearson(v, D):>10.4f}{spearman(v, D):>10.4f}')

print()
print('===== eff_density_ratio (方案A) 分布 =====')
R = np.array([r['ratio'] for r in rows])
for q in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
    print(f'  P{q:<3}= {np.percentile(R, q):.3f}')
print(f'  mean={R.mean():.3f} std={R.std():.3f}')

print()
print('===== eff_density_ratio 按定数段 =====')
bins = [('<13', 0, 13), ('13-14', 13, 14), ('14-15', 14, 15), ('15-16', 15, 16), ('16-17', 16, 17), ('>=17', 17, 99)]
for name, lo, hi in bins:
    sel = [r for r in rows if lo <= r['diff'] < hi]
    if not sel: continue
    rr = np.array([r['ratio'] for r in sel])
    dd = np.array([r['dens'] for r in sel])
    print(f'  {name:<6} n={len(sel):<4} ratio mean={rr.mean():.3f} P25={np.percentile(rr,25):.3f} P75={np.percentile(rr,75):.3f} | dens mean={dd.mean():.2f}')

print()
print('===== ratio 与多押特征相关性 (低ratio应=多押撑密度) =====')
print(f'  ratio vs mf3:       pearson={pearson(R, [r["mf3"] for r in rows]):.4f} spearman={spearman(R, [r["mf3"] for r in rows]):.4f}')
print(f'  ratio vs mf4:       pearson={pearson(R, [r["mf4"] for r in rows]):.4f} spearman={spearman(R, [r["mf4"] for r in rows]):.4f}')
print(f'  ratio vs wmf:       pearson={pearson(R, [r["wmf"] for r in rows]):.4f} spearman={spearman(R, [r["wmf"] for r in rows]):.4f}')
print(f'  dens vs mf3:        pearson={pearson([r["dens"] for r in rows], [r["mf3"] for r in rows]):.4f}')
print(f'  eff_avg vs mf3:     pearson={pearson([r["effa"] for r in rows], [r["mf3"] for r in rows]):.4f}')

print()
print('===== ratio 最低的 12 张官谱 (多押撑密度嫌疑) =====')
for r in sorted(rows, key=lambda x: x['ratio'])[:12]:
    print(f'  {r["name"][:40]:<42} diff={r["diff"]:>5.1f} dens={r["dens"]:>5.1f} effa={r["effa"]:>5.2f} ratio={r["ratio"]:>5.2f} mf3={r["mf3"]:>5.0f}')

print()
print('===== ratio 最高的 12 张官谱 (真单指连打) =====')
for r in sorted(rows, key=lambda x: -x['ratio'])[:12]:
    print(f'  {r["name"][:40]:<42} diff={r["diff"]:>5.1f} dens={r["dens"]:>5.1f} effa={r["effa"]:>5.2f} ratio={r["ratio"]:>5.2f} mf3={r["mf3"]:>5.0f}')

# 保存中间结果供方案B使用
with open(os.path.join(_ROOT, 'tools', '_tmp_ratio_analysis.pkl'), 'wb') as f:
    pickle.dump(rows, f)
print()
print('已保存中间结果 tools/_tmp_ratio_analysis.pkl')
