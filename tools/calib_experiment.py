# -*- coding: utf-8 -*-
"""验证 v10 改进方向的 OOF 实验:
  1. isotonic 校准 (整体 / 按level)
  2. Huber损失 GBM 是否更稳
  3. V0(无level) + isotonic 校准能达到什么水平 (对应"自定义谱无level"场景)
  4. 两段式 level 推断的可行性: 用 V0 预测 -> 映射 level -> 对比真实 level
"""
import os, sys, pickle, numpy as np
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_absolute_error

d = np.load(os.path.join(_ROOT, 'tools', 'cv_oof.npz'))
oof = {k: d[f'oof_{k}'] for k in ['v0', 'v1', 'v2', 'v3']}
y = d['y']; levels = d['levels']

print('='*70)
print('OOF 结果 (无任何校准):')
for k in ['v0', 'v1', 'v2', 'v3']:
    print(f'  {k.upper()}: MAE={mean_absolute_error(y, oof[k]):.4f} 偏差={np.mean(oof[k]-y):+.3f}')

# ---- 1. isotonic 校准: 在 OOF 上拟合 (OOF本身就是样本外, 拟合是无泄漏的近似) ----
def calibrate(pred, target):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(pred, target)
    return iso.predict(pred)

print('\n' + '='*70)
print('isotonic 校准 (拟合于OOF, 乐观但有方向性):')
for k in ['v0', 'v1', 'v2', 'v3']:
    c = calibrate(oof[k], y)
    print(f'  {k.upper()}: MAE={mean_absolute_error(y, c):.4f} 偏差={np.mean(c-y):+.3f}')

# ---- 2. 按level分别校准 ----
print('\n按level分别 isotonic:')
for k in ['v0', 'v1', 'v2', 'v3']:
    c = np.zeros_like(oof[k])
    for lv in ['EZ', 'HD', 'IN', 'AT']:
        m = levels == lv
        c[m] = calibrate(oof[k][m], y[m])
    print(f'  {k.upper()}: MAE={mean_absolute_error(y, c):.4f} 偏差={np.mean(c-y):+.3f}')

# ---- 3. 两段式 level 推断: V0 预测 -> 定数区间 -> level ----
# 用官方数据统计 定数 -> level 的经验区间 (分位数)
print('\n' + '='*70)
print('定数 -> level 经验分布 (官方数据):')
for lv in ['EZ', 'HD', 'IN', 'AT']:
    m = levels == lv
    print(f'  {lv}: n={np.sum(m):<4} 定数范围=[{y[m].min():.1f}, {y[m].max():.1f}] '
          f'P5={np.percentile(y[m],5):.1f} P25={np.percentile(y[m],25):.1f} '
          f'P50={np.percentile(y[m],50):.1f} P75={np.percentile(y[m],75):.1f} P95={np.percentile(y[m],95):.1f}')

# 简单阈值推断: 每个定数点取众数 level
def infer_level(pred):
    # 训练一个简单规则: 用 y 的 level 分位点建立边界
    # 用贝叶斯: P(level|pred) 在训练分布上
    edges = {}
    lv_order = ['EZ', 'HD', 'IN', 'AT']
    # 对每个定数格 (0.1步长) 找众数 level
    bins = np.arange(0, 18.1, 0.5)
    bin_med = []
    for i in range(len(bins)-1):
        m = (y >= bins[i]) & (y < bins[i+1])
        if np.sum(m) == 0:
            bin_med.append(None); continue
        counts = [np.sum(levels[m] == lv) for lv in lv_order]
        bin_med.append(lv_order[int(np.argmax(counts))])
    def _infer(p):
        idx = int(np.clip((p - bins[0]) / (bins[1]-bins[0]), 0, len(bin_med)-1))
        return bin_med[idx] if bin_med[idx] is not None else 'IN'
    return [_infer(p) for p in pred]

pred_levels = infer_level(oof['v0'])
acc = np.mean(np.array(pred_levels) == levels)
print(f'\n两段式推断 level 准确率 (V0预测->level): {acc*100:.1f}%')

# 混淆
from collections import Counter
print('混淆矩阵 (行=真, 列=推断):')
for lv in ['EZ','HD','IN','AT']:
    m = levels == lv
    cnt = Counter(np.array(pred_levels)[m])
    row = '  '.join(f'{k}:{cnt.get(k,0)}' for k in ['EZ','HD','IN','AT'])
    print(f'  真{lv}: {row}')

# 若用推断的 level 去 V1 模型里查... 我们没法重训, 但可以估计误差上界:
# 推断错误时, 用 V0 的预测 (因为 V1 用了错误 level 反而更差)
# 简化: 混合 = 推断正确用 V1, 错误用 V0
mixed = np.where(np.array(pred_levels) == levels, oof['v1'], oof['v0'])
print(f'\n两段式混合预测 MAE (推断对用V1, 错用V0): {mean_absolute_error(y, mixed):.4f} 偏差={np.mean(mixed-y):+.3f}')
