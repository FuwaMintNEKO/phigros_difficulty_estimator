# -*- coding: utf-8 -*-
"""v13.0 社区共识17+定数表预测导出: manifest17plus.tsv -> v130d_manifest17plus_predictions.csv
- 本地json提取特征预测 (Aegleseeker负id用test_charts的pez)
- 无谱面的行 status=无谱面
"""
import os, sys, io, csv, zipfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
from unified_parser import load_chart_from_bytes

# 训练/留出标注 (留出50首来自 v13_aux)
try:
    import pickle as _pk
    _aux = _pk.load(open(os.path.join(_ROOT, 'models', 'v13_aux.pkl'), 'rb'))
    HOLD_IDS = {_aux['hold_names'][i][0] for i in range(len(_aux['hold_names']))}
except Exception:
    HOLD_IDS = set()

rows = []
for l in open(os.path.join(_ROOT, 'data', 'phira', 'manifest17plus.tsv'), encoding='utf-8', errors='replace').read().splitlines()[2:]:
    p = l.split('	')
    if len(p) < 3 or not p[0].lstrip('-').isdigit():
        continue
    cid = int(p[0])
    name, diff = p[1], float(p[2])
    cd = None
    status = ''
    if cid == -2:   # Aegleseeker (表内负id, 用户给的pez测试文件)
        try:
            z = zipfile.ZipFile(os.path.join(_ROOT, 'data', 'test_charts', 'Aegleseeker.pez'))
            cd, _ = load_chart_from_bytes(z.read('64218553.json'))
            status = 'pez'
        except Exception:
            status = 'pez读取失败'
    elif cid < 0:
        status = '无谱面(已下架)'   # 负id除-2外均为下架谱, 本地同名id非同一谱
    else:
        for d in ['json_unranked_4star', 'json_unranked', 'json']:
            path = os.path.join(_ROOT, 'data', 'phira', d, '%d.json' % abs(cid))
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        raw = f.read()
                    cd, _ = load_chart_from_bytes(raw)
                    status = 'ok'
                except Exception as e:
                    status = '解析失败:' + str(e)[:40]
                break
        if cd is None and status == '':
            status = '无谱面'
    role = '留出' if abs(cid) in HOLD_IDS else ('训练' if cid > 0 else '测试')
    if cd is None:
        rows.append([cid, name, diff, '', '', status, role])
        continue
    try:
        r, err = app_mod.predict_one_chart(cd, speed=1.0, level='AT', is_custom=True)
        if r is None:
            rows.append([cid, name, diff, '', '', '预测失败:' + str(err)[:40], role])
        else:
            pr = round(r['prediction'], 2)
            rows.append([cid, name, diff, pr, round(pr - diff, 2), status, role])
    except Exception as e:
        rows.append([cid, name, diff, '', '', '异常:' + str(e)[:40], role])

out = os.path.join(_ROOT, 'data', 'phira', 'v130d_manifest17plus_predictions.csv')
with open(out, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'name', 'diff(社区共识表)', 'pred', 'err', 'status', 'role'])
    w.writerows(rows)
ok = sum(1 for r in rows if r[3] != '')
print('共 %d 行, 预测成功 %d, 输出 %s' % (len(rows), ok, out))
for r in rows:
    if r[3] == '':
        print('  无预测: #%s %s (%s)' % (r[0], r[1][:30], r[5]))
