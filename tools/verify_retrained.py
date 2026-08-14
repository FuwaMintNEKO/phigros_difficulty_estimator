# -*- coding: utf-8 -*-
"""用当前 pkl 模型预测关键自制谱 (重训后验证)"""
import os, sys, re
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import app
from unified_parser import load_chart_from_bytes

DL = r'C:\Users\NaNK\Downloads'
PAT = re.compile(r'^(.*?)\((\d+(?:\.\d+)?)(?:~(\d+(?:\.\d+)?))?\)(?:\(\d+\))?[^.]*\.json$')

def level_for(d):
    if d is None: return 'AT'
    if d >= 16.5: return 'AT'
    if d >= 11.5: return 'IN'
    if d >= 6.5: return 'HD'
    return 'EZ'

print(f'模型版本: {app.VERSION}')
print(f'boost特征数: {len(app.MANUAL_FLAT)}, caps: {app.CAPS}')
print()

# 关键无定数谱
KEY = ['Retribution_FULL.json', 'Sigma (Haocore Mix) ~ Regrets of The Yellow Tuli.json',
       'ボーカルに無茶させんな.json', '♿Unlimited Hyperlink♿.json']
print(f'{"谱面":<32} {"level":>5} {"pred":>7}')
for fn in KEY:
    p = os.path.join(DL, fn)
    if not os.path.exists(p): continue
    with open(p, 'rb') as f:
        cd, _ = load_chart_from_bytes(f.read())
    res, err = app.predict_one_chart(cd, speed=1.0, level='AT')
    if res:
        print(f'{fn[:32]:<32} {"AT":>5} {res["prediction"]:>7.2f}')
    else:
        print(f'{fn[:32]:<32} err={err}')

# 有定数自制谱
print()
print(f'{"谱面":<32} {"定数":>6} {"pred":>7} {"偏差":>7}')
for fn in sorted(os.listdir(DL)):
    if not fn.lower().endswith('.json'):
        continue
    m = PAT.match(fn)
    if m:
        name, a, b = m.group(1), float(m.group(2)), m.group(3)
        ud = (a + float(b)) / 2 if b else a
    else:
        continue
    p = os.path.join(DL, fn)
    try:
        with open(p, 'rb') as f:
            cd, _ = load_chart_from_bytes(f.read())
        res, err = app.predict_one_chart(cd, speed=1.0, level=level_for(ud))
        if res:
            print(f'{name[:32]:<32} {ud:>6.1f} {res["prediction"]:>7.2f} {res["prediction"]-ud:>+7.2f}')
    except Exception as e:
        pass
