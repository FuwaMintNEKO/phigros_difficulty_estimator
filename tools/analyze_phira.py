# -*- coding: utf-8 -*-
"""分析 phira 自制谱: 模型预测 vs 社区定数

- 读取 data/phira/json/{id}.json (RPE格式) → 解析 → 预测
- 与 charts.json 里的社区定数对比
- 按定数分段输出偏差, 检验"社区普遍高估"假设
- 输出: data/phira/predictions.csv + 终端摘要
"""
import os, sys, json, csv, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from unified_parser import load_chart_from_bytes
import app

CHART_META = os.path.join(_ROOT, 'data', 'phira', 'charts.json')
JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json')


def meta_map():
    charts = json.load(open(CHART_META, encoding='utf-8'))
    m = {}
    for lst in charts.values():
        for c in lst:
            m[c['id']] = c
    return m


def real_name(cd):
    meta = cd.get('META') or {}
    return meta.get('name') or meta.get('song') or None


def main():
    meta = meta_map()
    rows = []
    fails = []
    for fn in sorted(os.listdir(JSON_DIR)):
        if not fn.endswith('.json'):
            continue
        cid = int(fn[:-5])
        info = meta.get(cid, {})
        path = os.path.join(JSON_DIR, fn)
        try:
            with open(path, 'rb') as f:
                raw = f.read()
            cd, pe = load_chart_from_bytes(raw)
            if cd is None:
                fails.append((cid, 'parse None'))
                continue
            # 双 level 预测 (IN / AT) + 按社区定数映射的主 level
            lv = 'AT' if info.get('difficulty', 0) >= 16.5 else 'IN'
            is_custom = app.is_custom_chart(cd, pe)
            r, e = app.predict_one_chart(cd, speed=1.0, level=lv, is_custom=is_custom)
            if r is None:
                fails.append((cid, e))
                continue
            r_in, _ = app.predict_one_chart(cd, speed=1.0, level='IN', is_custom=is_custom)
            r_at, _ = app.predict_one_chart(cd, speed=1.0, level='AT', is_custom=is_custom)
            rows.append({
                'id': cid,
                'name': real_name(cd) or info.get('name', ''),
                'meta_name': info.get('name', ''),
                'diff': info.get('difficulty', 0),
                'level': info.get('level', ''),
                'ranked': info.get('ranked'),
                'pred': r['prediction'], 'gb': r['gb'], 'boost': r['boost'],
                'pred_in': r_in['prediction'] if r_in else None,
                'pred_at': r_at['prediction'] if r_at else None,
                'notes': r.get('total_notes'), 'dur': r.get('duration_sec'),
            })
        except Exception as ex:
            fails.append((cid, str(ex)[:80]))

    # 保存 CSV
    csv_path = os.path.join(_ROOT, 'data', 'phira', 'predictions.csv')
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [], extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    print(f'解析成功 {len(rows)}, 失败 {len(fails)}')
    for cid, e in fails[:10]:
        print(f'  FAIL {cid}: {e}')

    if not rows:
        return

    print()
    print(f'{"谱面名":<26} {"社区定数":>7} {"lv":>5} {"预测":>6} {"偏差":>7} {"pred_in":>8} {"pred_at":>8} {"notes":>6}')
    print('-' * 100)
    for r in sorted(rows, key=lambda x: -x['diff']):
        bias = r['pred'] - r['diff']
        print(f'{str(r["name"])[:26]:<26} {r["diff"]:>7.1f} {str(r["level"])[:5]:>5} '
              f'{r["pred"]:>6.2f} {bias:>+7.2f} {r["pred_in"]:>8.2f} {r["pred_at"]:>8.2f} {r["notes"]:>6}')

    # 分桶统计
    print()
    print('=== 按社区定数分桶偏差 ===')
    buckets = [(16.5, 99, '>=16.5'), (14, 16.5, '14-16.5'), (11, 14, '11-14'), (0, 11, '<11')]
    import statistics
    for lo, hi, label in buckets:
        grp = [r for r in rows if lo <= r['diff'] < hi]
        if not grp:
            continue
        biases = [r['pred'] - r['diff'] for r in grp]
        print(f'{label:<10} n={len(grp):>3}  均值偏差={statistics.mean(biases):>+7.3f}  MAE={statistics.mean(abs(b) for b in biases):.3f}  '
              f'高估谱数={sum(1 for b in biases if b>0.3)} 低估谱数={sum(1 for b in biases if b<-0.3)}')
    # 官方谱对照: 同特征区间下官方谱的定数分布(由训练集 OOF 已验证, 此处只报告自制谱整体)
    all_bias = [r['pred'] - r['diff'] for r in rows]
    print(f'全部: n={len(rows)} 均值偏差={statistics.mean(all_bias):+.3f} MAE={statistics.mean(abs(b) for b in all_bias):.3f}')


if __name__ == '__main__':
    main()
