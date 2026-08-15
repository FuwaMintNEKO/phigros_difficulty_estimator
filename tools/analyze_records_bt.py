# -*- coding: utf-8 -*-
"""A计划验证: 用玩家游玩记录做 Bradley-Terry 难度排序
输入: data/phira/records_hi/*.json (每谱至多300条记录)
方法:
  1. 玩家-谱面 best acc 矩阵
  2. 共享玩家配对: acc_A < acc_B → A更难; 平局各半票
  3. BT MM迭代 (s_i=难度), 只保留共享对局>=5的谱对
  4. 对照: top20玩家中位acc / 我们pred / 社区diff
"""
import os, sys, json, io, math, glob
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
RDIR = os.path.join(_ROOT, 'data', 'phira', 'records_hi')

charts = []
for f in sorted(glob.glob(os.path.join(RDIR, '*.json'))):
    d = json.load(open(f, encoding='utf-8'))
    charts.append(d)
print('谱数: %d' % len(charts))

# 1. 玩家 best acc 矩阵
player_acc = {}   # player -> {chart: best_acc}
for d in charts:
    cid = d['chart']
    for r in d['records']:
        p = r.get('player')
        a = r.get('accuracy')
        if p is None or a is None:
            continue
        m = player_acc.setdefault(p, {})
        if cid not in m or a > m[cid]:
            m[cid] = a

cids = [d['chart'] for d in charts]
n = len(cids)
idx = {c: i for i, c in enumerate(cids)}

# 2. 配对计数
W = np.zeros((n, n))   # W[i,j]: 玩家在 i 表现差于 j 的次数 (i更难)
Nij = np.zeros((n, n))
for m in player_acc.values():
    cs = [c for c in m if c in idx]
    for a in range(len(cs)):
        for b in range(a + 1, len(cs)):
            ia, ib = idx[cs[a]], idx[cs[b]]
            acc_a, acc_b = m[cs[a]], m[cs[b]]
            Nij[ia, ib] += 1
            Nij[ib, ia] += 1
            if acc_a < acc_b - 1e-9:
                W[ia, ib] += 1
            elif acc_b < acc_a - 1e-9:
                W[ib, ia] += 1
            else:
                W[ia, ib] += 0.5
                W[ib, ia] += 0.5

# 3. BT MM 迭代 (只统计共享对局>=5)
valid = Nij >= 5
active = [i for i in range(n) if np.sum(valid[i]) >= 3]   # 至少与3张谱有足够对局
print('有效谱(>=3张谱有共享对局>=5): %d / %d' % (len(active), n))
s = np.ones(n)
for it in range(300):
    s_new = np.zeros(n)
    for i in active:
        denom = 0.0
        for j in active:
            if valid[i, j]:
                denom += Nij[i, j] / (s[i] + s[j])
        if denom > 0:
            s_new[i] = np.sum(W[i, active] * valid[i, active]) / denom
        else:
            s_new[i] = s[i]
    s_new /= np.mean(s_new[active])
    if np.max(np.abs(s_new - s)) < 1e-7:
        s = s_new
        break
    s = s_new
bt = s / np.mean(s[active])

# 4. top20 中位 acc (高手视角)
top_acc = {}
for d in charts:
    accs = sorted((r.get('accuracy') for r in d['records'] if r.get('accuracy') is not None), reverse=True)
    top_acc[d['chart']] = float(np.median(accs[:20])) if accs else None

# 5. 读我们预测 + 社区定数
import csv
pred = {}
diff = {}
rows = list(csv.reader(open(os.path.join(_ROOT, 'data', 'phira', 'v1210_unranked_4star_predictions.csv'), encoding='utf-8-sig')))
for r in rows[1:]:
    try:
        pred[int(r[0])] = float(r[4])
        diff[int(r[0])] = float(r[2])
    except Exception:
        pass
names = {}
import json as _j
ua = _j.load(open(os.path.join(_ROOT, 'data', 'phira', 'unranked_all.json'), encoding='utf-8'))
for c in ua:
    names[c['id']] = c['name']

# 6. 输出
order = sorted(active, key=lambda i: -bt[i])
print()
print('%-6s %-26s %6s %6s %6s %6s %8s' % ('id', '谱名', 'BT难', 'top20acc', '我们pred', '社区diff', '对局数'))
for i in order:
    c = cids[i]
    print('%-6d %-26s %6.3f %6.4f %6.2f %6.1f %8d' % (
        c, (names.get(c) or '?')[:26], bt[i], top_acc.get(c) or -1, pred.get(c, -1), diff.get(c, -1), int(np.sum(valid[i]))))
print()
# 与 pred 相关
btv = np.array([bt[i] for i in active])
pv = np.array([pred.get(cids[i], -1) for i in active])
ok = pv > 0
rho = np.corrcoef(btv[ok], pv[ok])[0, 1]
print('BT难度 vs 我们pred: spearman=%.3f (n=%d)' % (np.corrcoef(np.argsort(np.argsort(btv[ok])), np.argsort(np.argsort(pv[ok])))[0,1], ok.sum()))
print('BT难度 vs 我们pred: pearson=%.3f' % rho)
# 分歧排行: 大众(acc)认为更难/更简单
print()
bt_rank = np.empty(n); bt_rank[active] = np.argsort(np.argsort(-btv))
pred_rank = np.empty(n); pred_rank[active] = np.argsort(np.argsort(-pv))
pv_map = np.full(n, -1.0); pv_map[active] = pv
print('--- 大众觉得更难(排名差 = 大众排名 - 我们排名, 正值=我们低估) ---')
diff_rank = [(i, bt_rank[i] - pred_rank[i]) for i in active if pv_map[i] > 0]
for i, dr in sorted(diff_rank, key=lambda x: -x[1])[:10]:
    print('  #%-6d %-26s pred=%5.2f 大众BT排名%2d 我们排名%2d 差%+3d top20acc=%.4f' % (
        cids[i], (names.get(cids[i]) or '?')[:26], pv[i], int(bt_rank[i]) + 1, int(pred_rank[i]) + 1, int(dr), top_acc.get(cids[i]) or -1))
print('--- 大众觉得更简单(负值=我们高估) ---')
for i, dr in sorted(diff_rank, key=lambda x: x[1])[:10]:
    print('  #%-6d %-26s pred=%5.2f 大众BT排名%2d 我们排名%2d 差%+3d top20acc=%.4f' % (
        cids[i], (names.get(cids[i]) or '?')[:26], pv[i], int(bt_rank[i]) + 1, int(pred_rank[i]) + 1, int(dr), top_acc.get(cids[i]) or -1))
