# -*- coding: utf-8 -*-
"""验证: 备份的原始v10模型 + 当前代码能否复现 Chart_SP #1347 AT=17.81
并检查备份pkl的字段 (MANUAL_FLAT/caps/FN) 与 Chart_SP 的谱面格式
"""
import os, sys, pickle, json
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 1. 备份 pkl 字段
with open(os.path.join(_ROOT, 'models', '6dim_model_v10_backup_old.pkl'), 'rb') as f:
    m = pickle.load(f)
print('=== 备份pkl(原始v10) ===')
print('keys:', sorted(m.keys()))
print('version:', m.get('version'), ' n_train:', m.get('n_train'))
print('FN数量:', len(m.get('feature_names', [])))
print('MANUAL_FLAT:', m.get('MANUAL_FLAT'))
print('caps:', m.get('caps'))
print()

# 2. Chart_SP #1347 格式
fn = os.path.join(r'C:\Users\NaNK\Downloads', 'Chart_SP #1347(1).json')
with open(fn, 'r', encoding='utf-8', errors='ignore') as f:
    data = json.load(f)
meta = data.get('META', {})
print('=== Chart_SP #1347 ===')
print('META keys:', list(meta.keys())[:15])
print('RPEVersion:', meta.get('RPEVersion'))
print('numOfNotes:', data.get('numOfNotes'))
print('judgeLine数:', len(data.get('judgeLineList', [])))
# 顶层 speedEvents 字段样例
for jl in data.get('judgeLineList', [])[:2]:
    se = jl.get('speedEvents', [])
    if se:
        print('speedEvents样例:', json.dumps(se[0], ensure_ascii=False)[:150])
        break
