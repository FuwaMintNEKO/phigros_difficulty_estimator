# -*- coding: utf-8 -*-
"""上架谱 (type=0) 预测定数 vs 社区定数 对比

- 只统计上架分区、difficulty>0 的谱
- 输出: 分桶汇总 + 高估/低估 top + 完整 CSV (data/phira/ranked_compare.csv)
"""
import os, sys, json, csv, io, statistics
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

charts = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8'))
ranked_ids = {c['id'] for c in charts['上架']}
print(f'上架谱总数: {len(ranked_ids)} (含 difficulty=0 的未评级谱)')

rows = []
for r in csv.DictReader(open(os.path.join(_ROOT, 'data', 'phira', 'predictions.csv'), encoding='utf-8-sig')):
    cid = int(r['id'])
    if cid not in ranked_ids:
        continue
    d = float(r['diff'])
    if d <= 0:
        continue
    rows.append(r)
print(f'上架且有定数: {len(rows)}')

# 分桶
buckets = [(17, 99, '>=17'), (16.5, 17, '16.5-17'), (16, 16.5, '16-16.5'), (15, 16, '15-16'),
           (14, 15, '14-15'), (12, 14, '12-14'), (10, 12, '10-12'), (0, 10, '<10')]
print(f'\n{"桶":<10} {"n":>4} {"均值偏差":>9} {"MAE":>7} {"高估>0.3":>8} {"低估<-0.3":>8}')
for lo, hi, label in buckets:
    grp = [r for r in rows if lo <= float(r['diff']) < hi]
    if not grp:
        continue
    bs = [float(r['pred']) - float(r['diff']) for r in grp]
    print(f'{label:<10} {len(grp):>4} {statistics.mean(bs):>+9.3f} {statistics.mean(abs(b) for b in bs):>7.3f} '
          f'{sum(1 for b in bs if b > 0.3):>8} {sum(1 for b in bs if b < -0.3):>8}')

all_b = [float(r['pred']) - float(r['diff']) for r in rows]
print(f'\n全部: n={len(rows)} 均值偏差={statistics.mean(all_b):+.3f} MAE={statistics.mean(abs(b) for b in all_b):.3f}')

print(f'\n{"谱面名":<30} {"社区":>5} {"预测":>6} {"偏差":>7} {"GB":>6} {"boost":>6} {"notes":>6}')
print('-' * 84)
for r in sorted(rows, key=lambda x: float(x['pred']) - float(x['diff'])):
    d, p = float(r['diff']), float(r['pred'])
    print(f'{str(r["name"])[:30]:<30} {d:>5.1f} {p:>6.2f} {p-d:>+7.2f} '
          f'{float(r["gb"]):>6.2f} {float(r["boost"]):>6.2f} {r["notes"]:>6}')

# 保存 CSV
with open(os.path.join(_ROOT, 'data', 'phira', 'ranked_compare.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'meta_name', 'name', 'diff', 'level', 'ranked', 'pred', 'gb', 'boost',
                                      'pred_in', 'pred_at', 'notes', 'dur'])
    w.writeheader()
    w.writerows(rows)
print(f'\n已保存 data/phira/ranked_compare.csv')
