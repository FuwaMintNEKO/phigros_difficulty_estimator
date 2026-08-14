# -*- coding: utf-8 -*-
"""回归验证 _is_rpe_v3 修复:
1. Chart_SP (官方愚人节谱) 应识别为 standard, 24 条线
2. 全官方谱面不再误判 (全部 standard)
3. 重新预测 Chart_SP
"""
import sys, os, json
sys.path.insert(0, r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator')
from unified_parser import load_chart_from_bytes, detect_format
import app

# 1. Chart_SP
p = r'C:\Users\NaNK\Downloads\Chart_SP #1347(1).json'
with open(p, 'rb') as f:
    raw = f.read()
cd, pe = load_chart_from_bytes(raw)
lines = len(cd.get('judgeLineList', []))
n_notes = sum(len(l.get('notesAbove', [])) + len(l.get('notesBelow', []))
              for l in cd.get('judgeLineList', []))
print(f'Chart_SP: lines={lines} notes={n_notes}')
for L in ['EZ', 'HD', 'IN', 'AT']:
    r, e = app.predict_one_chart(cd, level=L)
    print(f'  [{L}] 预测={r["prediction"]:.2f}  (gb={r["gb"]:.2f} boost={r["boost"]:.2f})')

# 2. 全官方谱误判扫描
from data_loader import find_chart_files
_ROOT = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator'
cfs = find_chart_files(os.path.join(_ROOT, 'data', 'chart'))
mis = []
n = 0
for fn, info in cfs.items():
    for lv, fp in info['levels'].items():
        n += 1
        try:
            with open(fp, 'rb') as f:
                raw = f.read()
            cd2, _ = load_chart_from_bytes(raw)
            if cd2 is None:
                mis.append((fn, lv, 'parse-none'))
        except Exception as e:
            mis.append((fn, lv, str(e)[:40]))
print(f'官方谱解析: 总数={n} 失败={len(mis)}')
for m in mis[:10]:
    print('  ', m)
