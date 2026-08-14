# -*- coding: utf-8 -*-
"""V14 (v10) 模型对本地全部官谱 JSON 的批量预测，与 difficulty.tsv 对比。
走 app.predict_one_chart 生产路径（level onehot + boost，官方谱不应用 density align）。
"""
import os, sys, json, csv
import numpy as np

ROOT = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator'
sys.path.insert(0, ROOT)

from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
import app

diffs = load_difficulty_tsv(os.path.join(ROOT, 'data', 'info', 'difficulty.tsv'))
chart_files = find_chart_files(os.path.join(ROOT, 'data', 'chart'))

rows = []
for folder, info in chart_files.items():
    sid = info['song_id']
    if sid not in diffs:
        continue
    for lv, path in info['levels'].items():
        if lv not in diffs[sid]:
            continue
        try:
            cd = load_chart_json(path)
            res, err = app.predict_one_chart(cd, speed=1.0, level=lv)
            if not res:
                continue
            pred = res['prediction']
            real = diffs[sid][lv]
            rows.append({
                'song': sid, 'folder': folder, 'lv': lv,
                'real': real, 'pred': pred, 'diff': pred - real,
                'gb': res['gb'], 'boost': res['boost'],
                'nps': res['real_notes_per_second'], 'dur': res['duration_sec'],
                'bpm': res['bpm'],
            })
        except Exception as e:
            print(f'  [跳过] {folder}/{lv}: {e}')

rows.sort(key=lambda r: r['real'])
errs = np.array([r['diff'] for r in rows])
reals = np.array([r['real'] for r in rows])

print(f'总计 {len(rows)} 个谱面')
print(f'整体: MAE={np.mean(np.abs(errs)):.3f}  Bias={np.mean(errs):+.3f}  median={np.median(errs):+.3f}')

# 按定数分段
bins = [(1,4),(4,7),(7,9),(9,11),(11,13),(13,14.5),(14.5,16),(16,17),(17,20)]
print('按定数分段:')
for lo, hi in bins:
    mask = (reals >= lo) & (reals < hi)
    if mask.sum() == 0:
        continue
    e = errs[mask]
    print(f'  [{lo:4.1f},{hi:4.1f}): n={mask.sum():4d}  Bias={np.mean(e):+.3f}  MAE={np.mean(np.abs(e)):.3f}')

# 按 level
print('按 level:')
for lv in ['EZ', 'HD', 'IN', 'AT']:
    mask = np.array([r['lv'] == lv for r in rows])
    if mask.sum() == 0:
        continue
    e = errs[mask]
    print(f'  {lv}: n={mask.sum():4d}  Bias={np.mean(e):+.3f}  MAE={np.mean(np.abs(e)):.3f}')

# 重点曲目（社区讨论过的个例）
print('\n重点曲目:')
foci = ['dBdoll.YUESTEVENuen', 'ReEndofaDream.umavsモリモリあつし', 'volcanic.DETROakaルゼ',
        'QZKagoRequiem.tpazolite', 'DistortedFate.Sakuzyo', 'Igallta.SeURa', 'Rrharil.TeamGrimoire',
        'DESTRUCTION321.Normal1zervsBrokenNerdz', 'DerRichter.Ωμεγα', 'DerSchneid.Ωμεγα',
        '玩具狂奏曲終焉.きくお', '雪降りメリクリ.A39', 'ERABYECONNEC10N.かめりあ',
        'StardustRAY.kanonevsBlackY', 'Hydra.JamesLandinoXAkiraComplex',
        'BANGINGSTRIKE.DewPleiades', 'Aleph0.LeaF', 'Spasmodic.姜米條颶風元力上人',
        '狂喜蘭舞.LeaF', 'CROSSSOUL.HyuNfeatSyepias', 'NowIsTheTimeDoIt.RoyMikelate',
        'Burn.NceS', 'modulus.PTB10', 'もぺもぺ.LeaF', 'DistortedFate.Sakuzyo']
for r in rows:
    if r['song'] in foci:
        print(f'  {r["song"][:38]:38s} [{r["lv"]}] 真实={r["real"]:5.2f} 预测={r["pred"]:5.2f} d={r["diff"]:+.2f} '
              f'(gb={r["gb"]:.2f}+b={r["boost"]:.2f}) nps={r["nps"]:.2f} dur={r["dur"]:.0f}s')

# 重点曲目 AT 档的 boost 分类构成
print('\n重点曲目 AT 档 boost 分类构成:')
for r in rows:
    if r['song'] in foci and r['lv'] == 'AT':
        cd = load_chart_json(r['folder'] and next(p for l, p in chart_files[r['folder']]['levels'].items() if l == 'AT'))
        res2, _ = app.predict_one_chart(cd, speed=1.0, level='AT')
        cats = res2['categories']
        print(f'  {r["song"][:36]:36s} b={r["boost"]:.2f}  密度={cats.get("密度",0):.2f} 配置={cats.get("配置",0):.2f} 耐力={cats.get("耐力",0):.2f} 读谱={cats.get("读谱",0):.2f}')

# 最离谱 15 个
print('\n|误差| 最大 15 个:')
worst = sorted(rows, key=lambda r: -abs(r['diff']))[:15]
for r in worst:
    print(f'  {r["song"][:38]:38s} [{r["lv"]}] 真实={r["real"]:5.2f} 预测={r["pred"]:5.2f} d={r["diff"]:+.2f} '
          f'nps={r["nps"]:.2f} dur={r["dur"]:.0f}s')

# 存 CSV 供进一步分析
out = os.path.join(ROOT, 'data', 'phira', 'official_v14_predictions.csv')
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f'\n已保存: {out}')
