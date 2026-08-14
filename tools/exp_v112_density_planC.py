# -*- coding: utf-8 -*-
"""t2 方案C验证: eff_density_ratio 相对官谱基准残差 (去难度趋势)
residual = ratio - f(diff), f 从官谱拟合
验证: 残差是否预测上架谱偏差 (pred-diff)
"""
import os, sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
_ROOT = r'D:\\Trae项目\\新建文件夹\\phigros_difficulty_estimator'

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

# 从方案A中间结果加载
with open(os.path.join(_ROOT, 'tools', '_tmp_ratio_analysis.pkl'), 'rb') as f:
    rows = pickle.load(f)  # 官谱982 rows: name/level/diff/dens/effa/effp/rcnps/ratio/redun/mf3/mf4/wmf

off = [r for r in rows]
D = np.array([r['diff'] for r in off])
R = np.array([r['ratio'] for r in off])

# 官谱 ratio ~ diff 拟合 (分段线性近似: 用每段的官谱中位)
print('===== 官谱 ratio 基准 (按定数段中位) =====')
bins = [('<13', 0, 13), ('13-14', 13, 14), ('14-15', 14, 15), ('15-16', 15, 16), ('16-17', 16, 17), ('>=17', 17, 99)]
medians = {}
for name, lo, hi in bins:
    sel = [r['ratio'] for r in off if lo <= r['diff'] < hi]
    if sel:
        medians[name] = np.median(sel)
        print(f'  {name:<6} 中位={medians[name]:.3f}')

# 分段残差函数
def ratio_baseline(diff):
    if diff < 13: return medians.get('<13', 0.78)
    if diff < 14: return medians.get('13-14', 0.75)
    if diff < 15: return medians.get('14-15', 0.73)
    if diff < 16: return medians.get('15-16', 0.72)
    if diff < 17: return medians.get('16-17', 0.70)
    return medians.get('>=17', 0.60)

# 上架谱残差
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)

ranked = []
for r_ in cache['ranked']:
    if not r_.get('diff') or r_['diff'] <= 10: continue
    f = r_['feats']
    dens = f.get('above_avg_density_mean', 0)
    effa = f.get('eff_avg_tps_1s', 0)
    ratio = effa / max(dens, 0.1)
    diff = float(r_['diff'])
    ranked.append({'name': r_['name'], 'diff': diff, 'ratio': ratio,
                   'dens': dens, 'resid': ratio - ratio_baseline(diff),
                   'wmf': f.get('weighted_mf_score_per_sec', 0),
                   'mf3': f.get('multi_finger_3plus_events', 0),
                   'nps': f.get('notes_per_second', 0)})

off_resid = [r['ratio'] - ratio_baseline(r['diff']) for r in off]
rk_resid = [r['resid'] for r in ranked]

print()
print('===== 残差分布: 官谱 vs 上架谱 =====')
for label, arr in [('官谱', off_resid), ('上架', rk_resid)]:
    a = np.array(arr)
    print(f'  {label:<6} mean={a.mean():+.4f} P25={np.percentile(a,25):+.4f} P50={np.percentile(a,50):+.4f} P75={np.percentile(a,75):+.4f}')
print(f'  上架谱残差<0 (密度相对虚高) 占比: {np.mean(np.array(rk_resid)<0)*100:.1f}%')

# 残差与预测偏差关系: 加载方案B模拟的 pred_orig
with open(os.path.join(_ROOT, 'tools', '_tmp_planB_sim.pkl'), 'rb') as f:
    sim = pickle.load(f)
rk_sim = {r['name']: r for r in sim if r['src'] == 'ranked'}
bias_list, resid_list = [], []
for r in ranked:
    s = rk_sim.get(r['name'])
    if s:
        bias_list.append(s['pred_orig'] - r['diff'])
        resid_list.append(r['resid'])
print()
print(f'===== 残差 vs 预测偏差 (上架谱 n={len(bias_list)}) =====')
print(f'  pearson={pearson(resid_list, bias_list):.4f}  spearman={spearman(resid_list, bias_list):.4f}')
# 分组: 残差<0 (密度虚高) vs 残差>0.02 (真连打/密度扎实)
g_neg = [b for r_, b in zip(ranked, bias_list) if r_['resid'] < -0.02]
g_pos = [b for r_, b in zip(ranked, bias_list) if r_['resid'] > 0.02]
print(f'  残差<-0.02 (虚高): n={len(g_neg)} 平均偏差={np.mean(g_neg):+.3f}')
print(f'  残差>+0.02 (扎实): n={len(g_pos)} 平均偏差={np.mean(g_pos):+.3f}')

print()
print('===== 上架谱残差最小的 12 张 (密度最虚高, 需修正) =====')
for r in sorted(ranked, key=lambda x: x['resid'])[:12]:
    s = rk_sim.get(r['name'])
    bias = s['pred_orig'] - r['diff'] if s else float('nan')
    print(f'  {r["name"][:24]:<26} diff={r["diff"]:>5.1f} ratio={r["ratio"]:.3f} 残差={r["resid"]:+.3f} 偏差={bias:+.2f} dens={r["dens"]:.1f}')

print()
print('===== 残差最大的 12 张 (真连打/密度扎实) =====')
for r in sorted(ranked, key=lambda x: -x['resid'])[:12]:
    s = rk_sim.get(r['name'])
    bias = s['pred_orig'] - r['diff'] if s else float('nan')
    print(f'  {r["name"][:24]:<26} diff={r["diff"]:>5.1f} ratio={r["ratio"]:.3f} 残差={r["resid"]:+.3f} 偏差={bias:+.2f} dens={r["dens"]:.1f}')

with open(os.path.join(_ROOT, 'tools', '_tmp_planC_results.pkl'), 'wb') as f:
    pickle.dump({'ranked': ranked, 'off_resid': off_resid, 'baselines': medians}, f)
print()
print('已保存 tools/_tmp_planC_results.pkl')
