# -*- coding: utf-8 -*-
"""t4: 外推段(>=17.7)社区趋势 vs 模型预测分析
上架谱 6 张 + 特殊谱 1 张 (WACCA, ST Lv.FINAL) = 7 张
输出: logs/exp_extrapolation_analysis.txt
"""
import os, sys, json, csv, io
import numpy as np

_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

OUT = os.path.join(_ROOT, 'logs', 'exp_extrapolation_analysis.txt')
PRED_CSV = os.path.join(_ROOT, 'data', 'phira', 'predictions.csv')
CHART_META = os.path.join(_ROOT, 'data', 'phira', 'charts.json')
JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json')
THRESH = 17.7

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

def level_key(level_str):
    s = (level_str or '').upper()
    if 'AT' in s: return 'AT'
    if 'IN' in s: return 'IN'
    if 'HD' in s: return 'HD'
    return 'IN'  # 未知自定义level(ST/SP/FM/EX等)默认IN

def main():
    charts = json.load(open(CHART_META, encoding='utf-8'))
    # 标注来源: 上架/特殊
    src = {}
    meta_by_id = {}
    for lst_name in charts:
        for c in charts[lst_name]:
            meta_by_id[c['id']] = c
            src[c['id']] = lst_name

    with open(PRED_CSV, encoding='utf-8-sig', newline='') as f:
        pred_rows = list(csv.DictReader(f))
    by_id = {r['id']: r for r in pred_rows}

    # 找出 diff >= 17.7 的谱 (含上架+特殊, 都在 predictions.csv 中)
    hi = [r for r in pred_rows if float(r['diff']) >= THRESH]
    hi.sort(key=lambda r: -float(r['diff']))

    out = []
    out.append(_buf('=' * 100))
    out.append(_buf(f't4: 外推段(社区定数>={THRESH})模型预测 vs 社区定数分析'))
    out.append(_buf(f'官谱定数上限 17.6 (Rrharil AT); 自制谱 {THRESH}+ 为模型外推段'))
    out.append(_buf(''))

    # 1) 谱面列表
    out.append(_buf(f'--- 1) 社区定数>={THRESH} 的谱面 ({len(hi)} 张) ---'))
    out.append(_buf(f'{"id":<8} {"name":<32} {"level":<14} {"来源":<6} {"diff":>6} {"pred":>7} {"偏差":>7} {"gb":>8} {"boost":>7} {"tags"}'))
    rows_ok = []
    for r in hi:
        cid = r['id']
        meta = meta_by_id.get(int(cid), {})
        name = meta.get('name', r['name'])
        lv = meta.get('level', r['level'])
        diff = float(r['diff']); pred = float(r['pred'])
        bias = pred - diff
        tags = ','.join(meta.get('tags', []))
        out.append(_buf(f'{cid:<8} {name:<32} {lv:<14} {src.get(int(cid), "?"):<6} {diff:>6.2f} {pred:>7.4f} {bias:>+7.3f} {r["gb"]:>8} {r["boost"]:>7} {tags}'))
        rows_ok.append({'id': cid, 'name': name, 'level': lv, 'src': src.get(int(cid), '?'),
                        'diff': diff, 'pred': pred, 'bias': bias, 'gb': float(r['gb']), 'boost': float(r['boost']),
                        'tags': meta.get('tags', []), 'rating': meta.get('rating'), 'ratingCount': meta.get('ratingCount'),
                        'json_exists': os.path.exists(os.path.join(JSON_DIR, cid + '.json'))})

    # 2) 排序一致性
    out.append(_buf(''))
    out.append(_buf('--- 2) 排序一致性 (Spearman, 7张) ---'))
    ds = np.array([r['diff'] for r in rows_ok]); ps = np.array([r['pred'] for r in rows_ok])
    sp = spearman(ds, ps)
    pr = float(np.corrcoef(ds, ps)[0, 1]) if len(ds) >= 3 else float('nan')
    out.append(_buf(f'Spearman(社区定数 vs 模型预测) = {sp:.4f}   Pearson = {pr:.4f}'))
    order_rank = sorted(rows_ok, key=lambda r: r['diff'])
    out.append(_buf('按社区定数升序: ' + ' -> '.join(f"{r['name']}({r['diff']:.2f})" for r in order_rank)))
    out.append(_buf('按模型预测升序: ' + ' -> '.join(f"{r['name']}({r['pred']:.2f})" for r in sorted(rows_ok, key=lambda r: r['pred']))))
    # 逆序对数
    inv = sum(1 for i in range(len(ds)) for j in range(i+1, len(ds))
              if (ds[i] < ds[j]) != (ps[i] < ps[j]))
    out.append(_buf(f'逆序对数: {inv} / {len(ds)*(len(ds)-1)//2}'))

    # 3) 偏差模式
    out.append(_buf(''))
    out.append(_buf('--- 3) 偏差模式 ---'))
    hi_bias = [(r, r['bias']) for r in rows_ok if r['bias'] > 0.05]
    lo_bias = [(r, r['bias']) for r in rows_ok if r['bias'] < -0.05]
    out.append(_buf(f'高估(pred>diff+0.05): {len(hi_bias)} 张'))
    for r, b in sorted(hi_bias, key=lambda t: -t[1]):
        out.append(_buf(f'  +{b:.3f} {r["name"]} (diff {r["diff"]:.2f} -> pred {r["pred"]:.2f}) tags={r["tags"]} boost={r["boost"]:.3f}'))
    out.append(_buf(f'低估(pred<diff-0.05): {len(lo_bias)} 张'))
    for r, b in sorted(lo_bias, key=lambda t: t[1]):
        out.append(_buf(f'  {b:+.3f} {r["name"]} (diff {r["diff"]:.2f} -> pred {r["pred"]:.2f}) tags={r["tags"]} boost={r["boost"]:.3f}'))
    out.append(_buf(f'平均偏差: {np.mean([r["bias"] for r in rows_ok]):+.3f}   MAE: {np.mean([abs(r["bias"]) for r in rows_ok]):.3f}'))

    # 4) level 异常检查
    out.append(_buf(''))
    out.append(_buf('--- 4) level/解析异常检查 ---'))
    for r in rows_ok:
        mapped = level_key(r['level'])
        flag = ''
        if mapped != 'AT' and r['level'].upper().find('AT') >= 0 and mapped != 'AT':
            pass
        if 'ST' in r['level'].upper() or 'EX' in r['level'].upper() or 'FINAL' in r['level'].upper() or '?' in r['level']:
            flag = '  <-- 自定义level'
        out.append(_buf(f'  {r["name"]}: meta level={r["level"]!r} -> level_key映射={mapped} json存在={r["json_exists"]}{flag}'))
    # 官谱无 Lv.18/17.7+
    out.append(_buf('  注意: 官谱最高 AT Lv.17 (Rrharil 17.6), 17.7+ 均为自制谱扩展难度段'))

    # 5) 边界参考: 17.0~17.7 上架谱
    out.append(_buf(''))
    out.append(_buf('--- 5) 边界参考: 上架谱 diff 17.0~17.7 ---'))
    ref = []
    for r in pred_rows:
        d = float(r['diff'])
        if 17.0 <= d < THRESH and src.get(int(r['id'])) == '上架':
            meta = meta_by_id.get(int(r['id']), {})
            ref.append((d, float(r['pred']), meta.get('name', r['name']), meta.get('level', r['level']), r['id']))
    ref.sort(reverse=True)
    out.append(_buf(f'{"diff":>6} {"pred":>7} {"偏差":>7}  {"name":<32} {"level":<12} id'))
    for d, p, n, lv, cid in ref:
        out.append(_buf(f'{d:>6.2f} {p:>7.2f} {p-d:>+7.2f}  {n:<32} {lv:<12} {cid}'))
    out.append(_buf(''))
    # 外推趋势: diff 17.0-18.5 上架+特殊 全体偏差
    all_ext = [(float(r['diff']), float(r['pred'])) for r in pred_rows if float(r['diff']) >= 17.0]
    all_ext.sort()
    if len(all_ext) >= 3:
        out.append(_buf('--- 6) 外推段整体趋势 (diff>=17.0 全部谱) ---'))
        for d, p in all_ext:
            out.append(_buf(f'  diff {d:.2f} -> pred {p:.2f} (偏差 {p-d:+.2f})'))
        ds2 = np.array([x[0] for x in all_ext]); ps2 = np.array([x[1] for x in all_ext])
        out.append(_buf(f'  n={len(all_ext)} Spearman={spearman(ds2, ps2):.4f} Pearson={float(np.corrcoef(ds2, ps2)[0,1]):.4f} 平均偏差={np.mean(ps2-ds2):+.3f}'))

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(''.join(out))
    print(f'\n结果已写入: {OUT}')

if __name__ == '__main__':
    main()
