# -*- coding: utf-8 -*-
"""筛选 16+ 高评分 极端配置/被低估谱 (先用 v11.4 CSV 初筛, v11.5c 完成后更新)
标准: 社区定数>=16, rating>=0.85
输出: ① 预测低估(err<-0.4) ② 极端配置特征突出(多指/定轨/32分)
"""
import os, csv, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = r'D:\Trae项目\新建文件夹\phigros_difficulty_estimator'
rows = []
with open(os.path.join(_ROOT, 'data', 'phira', 'unranked_4star_list.csv'), encoding='utf-8-sig') as f:
    rd = csv.DictReader(f)
    for r_ in rd:
        try:
            d = float(r_['difficulty']) if r_['difficulty'] else None
            if d is None or d < 16: continue
            rt = float(r_['rating']); rc = int(r_['ratingCount'])
            if rt < 0.85: continue
            rows.append({'id': int(r_['id']), 'name': r_['name'], 'level': r_['level'], 'diff': d,
                         'rating': rt, 'rc': rc, 'pred': float(r_['pred']),
                         'mf3': float(r_['mf3']), 'mf4': float(r_['mf4']), 'dens': float(r_['dens']),
                         'eff_avg': float(r_['eff_avg']), 'nps': float(r_['nps'])})
        except Exception:
            pass
print(f'16+ 且 rating>=0.85: {len(rows)} 张')

# 按低估程度排序
rows.sort(key=lambda x: x['pred'] - x['diff'])
print('\n===== 被低估 top25 (预测-社区定数 最小) =====')
print(f'{"名称":<34}{"等级":<12}{"社区":>5}{"预测":>6}{"err":>7}{"rating":>7}{"票":>5}{"mf3":>5}{"dens":>5}')
for r in rows[:25]:
    print(f'{r["name"][:32]:<34}{r["level"][:10]:<12}{r["diff"]:>5.1f}{r["pred"]:>6.2f}{r["pred"]-r["diff"]:>+7.2f}{r["rating"]:>7.3f}{r["rc"]:>5}{r["mf3"]:>5.0f}{r["dens"]:>5.1f}')

# 极端配置突出: 多指 + 高评分
print('\n===== 纯多指谱 (mf3>=60) top15 =====')
mf_rows = [r for r in rows if r['mf3'] >= 60]
mf_rows.sort(key=lambda x: -x['rating'])
for r in mf_rows[:15]:
    print(f'{r["name"][:32]:<34}{r["level"][:10]:<12}{r["diff"]:>5.1f}{r["pred"]:>6.2f}{r["pred"]-r["diff"]:>+7.2f}{r["rating"]:>7.3f}{r["rc"]:>5}{r["mf3"]:>5.0f}{r["dens"]:>5.1f}')

# 多指+低密(被重压档, Cheerio同类)
print('\n===== 多指+低密 (mf3>=30 且 dens<9.5) 被低估 top15 =====')
sub = [r for r in rows if r['mf3'] >= 30 and r['dens'] < 9.5]
sub.sort(key=lambda x: x['pred'] - x['diff'])
for r in sub[:15]:
    print(f'{r["name"][:32]:<34}{r["level"][:10]:<12}{r["diff"]:>5.1f}{r["pred"]:>6.2f}{r["pred"]-r["diff"]:>+7.2f}{r["rating"]:>7.3f}{r["rc"]:>5}{r["mf3"]:>5.0f}{r["dens"]:>5.1f}')
print('DONE')
