# -*- coding: utf-8 -*-
"""特征审计: 官谱256特征的 冗余(共线/弱相关) 与 缺失(社区残差模式)"""
import os, sys, io, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

with open(os.path.join(_ROOT, 'models', '6dim_model_v11_4.pkl'), 'rb') as f:
    m = pickle.load(f)
FN = m['feature_names']
gb = m['gb']

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
off = cache['official']

# 1. GB 特征重要性 top30 (特征 + level onehot)
imp = gb.feature_importances_
LV = m.get('lv_order', ['EZ','HD','IN','AT'])
names_all = list(FN) + [f'level_{lv}' for lv in LV]
order = np.argsort(-imp)
print('=== GB 特征重要性 Top 35 ===')
for i in order[:35]:
    print(f'  {names_all[i]:<36} {imp[i]*100:5.2f}%')
print(f'  (level onehot合计: {imp[len(FN):].sum()*100:.1f}%)')

# 2. 特征-定数相关性 (官谱)
y = np.array([f['diff'] for f in off])
X = np.array([[f['feats'].get(n, 0) for n in FN] for f in off])
corrs = []
for j, n in enumerate(FN):
    if X[:, j].std() < 1e-9: continue
    r = np.corrcoef(X[:, j], y)[0, 1]
    corrs.append((n, r))
print('\n=== 弱相关特征 (|r|<0.05, 可能冗余) ===')
weak = [(n, r) for n, r in corrs if abs(r) < 0.05]
print(f'共 {len(weak)} 个:')
for n, r in sorted(weak, key=lambda x: abs(x[1]))[:25]:
    print(f'  {n:<36} r={r:+.3f}')

# 3. 共线性: 高相关特征对 (采样100特征)
print('\n=== 高共线特征对 (|r|>0.95, 抽样检测) ===')
n_feat = len(FN)
sample = list(range(0, n_feat, 2))  # 抽样一半
collinear = []
for a in range(len(sample)):
    for b in range(a+1, len(sample)):
        i, j = sample[a], sample[b]
        va, vb = X[:, i], X[:, j]
        if va.std() < 1e-9 or vb.std() < 1e-9: continue
        r = np.corrcoef(va, vb)[0, 1]
        if abs(r) > 0.95:
            collinear.append((FN[i], FN[j], r))
print(f'共线对: {len(collinear)}')
for a, b, r in sorted(collinear, key=lambda x: -abs(x[2]))[:20]:
    print(f'  {a:<32} ~ {b:<32} r={r:.3f}')
