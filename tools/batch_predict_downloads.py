# -*- coding: utf-8 -*-
"""批量预测 Downloads 下的自制谱 (文件名含用户自测定数)

用法: python tools/batch_predict_downloads.py
- 从文件名括号提取用户测定的定数, 如 "Apollo(17.8).json" -> 17.8
- 区间写法 "おぎゃりないざー(16.4~16.6).json" -> 取中值 16.5
- 按定数映射预测用的 level: >=16.5 -> AT, >=11.5 -> IN, >=6.5 -> HD, else EZ
- 同时输出该谱在 4 个 level 下的预测, 便于观察 level 敏感性
"""
import os, sys, re, json
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from unified_parser import load_chart_from_bytes
import app

DL = r'C:\Users\NaNK\Downloads'
# 匹配: 名字(定数).json / 名字(定数)(1).json / 名字(定数)_后缀.json / 名字(定数a~定数b).json
PAT = re.compile(r'^(.*?)\((\d+(?:\.\d+)?)(?:~(\d+(?:\.\d+)?))?\)(?:\(\d+\))?[^.]*\.json$')

def parse_label(fn):
    m = PAT.match(fn)
    if not m:
        return None
    name, a, b = m.group(1), float(m.group(2)), m.group(3)
    val = (a + float(b)) / 2 if b else a
    return name, val

def level_for(d):
    if d is None: return 'AT'  # 等级不明的谱(含 SP 谱)默认尝试 AT 预测
    if d >= 16.5: return 'AT'
    if d >= 11.5: return 'IN'
    if d >= 6.5: return 'HD'
    return 'EZ'

def real_name(cd):
    """从谱面数据提取真名: RPE 的 META.name / META.song; 没有则返回 None"""
    try:
        meta = cd.get('META') or {}
        return meta.get('name') or meta.get('song') or None
    except Exception:
        return None

rows = []
skipped = []
for fn in sorted(os.listdir(DL)):
    if not fn.lower().endswith('.json'):
        continue
    lbl = parse_label(fn)
    if lbl is None:
        # 无定数谱(含 SP 谱): 默认按 AT 预测, 便于用户比对
        user_d = None
        name = fn[:-5]
    else:
        name, user_d = lbl
    path = os.path.join(DL, fn)
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        cd, pe_text = load_chart_from_bytes(raw)
        if cd is None:
            skipped.append((fn, '解析失败'))
            continue
        rname = real_name(cd)
        lv = level_for(user_d)
        is_custom = app.is_custom_chart(cd, pe_text)
        res, err = app.predict_one_chart(cd, speed=1.0, level=lv, is_custom=is_custom)
        if res is None:
            skipped.append((fn, err))
            continue
        # 各 level 下的预测
        lv_preds = {}
        for L in ['EZ', 'HD', 'IN', 'AT']:
            r2, e2 = app.predict_one_chart(cd, speed=1.0, level=L, is_custom=is_custom)
            lv_preds[L] = r2['prediction'] if r2 else None
        rows.append({
            'file': fn, 'name': name, 'real': rname, 'user_d': user_d, 'level': lv,
            'pred': res['prediction'], 'gb': res['gb'], 'boost': res['boost'],
            'total_notes': res.get('total_notes'), 'duration': res.get('duration_sec'),
            'rcnps': res.get('real_core_notes_per_second'),
            'lv_preds': lv_preds,
        })
    except Exception as e:
        skipped.append((fn, str(e)[:80]))

print(f'{"谱面(真名)":<30} {"文件名":<22} {"定数":>6} {"lv":>3} {"预测":>7} {"偏差":>7} {"GB":>6} {"boost":>6} {"notes":>6} {"时长s":>6}')
print('-' * 118)
for r in sorted(rows, key=lambda x: (-x['user_d']) if x['user_d'] is not None else 999):
    disp = r['real'] or r['name']
    if r['user_d'] is None:
        # 无定数谱: 只显示 AT 预测 (用户要求 SP/等级不明默认 AT)
        at = r['lv_preds']['AT']
        print(f'{str(disp)[:30]:<30} {r["file"][:22]:<22} {"-":>6} {r["level"]:>3} {at:>7.2f} {"-":>7} '
              f'{r["gb"]:>6.2f} {r["boost"]:>6.2f} {r["total_notes"]:>6} {r["duration"]:>6.1f}  <- 无定数,AT预测')
    else:
        diff = r['pred'] - r['user_d']
        print(f'{str(disp)[:30]:<30} {r["file"][:22]:<22} {r["user_d"]:>6.1f} {r["level"]:>3} {r["pred"]:>7.2f} {diff:>+7.2f} '
              f'{r["gb"]:>6.2f} {r["boost"]:>6.2f} {r["total_notes"]:>6} {r["duration"]:>6.1f}')

print()
print('===== 各谱在所有 level 下的预测 (观察 level 敏感性) =====')
print(f'{"谱面":<24} {"定数":>6} {"EZ":>6} {"HD":>6} {"IN":>6} {"AT":>6}')
for r in rows:
    p = r['lv_preds']
    fmt = lambda v: f'{v:.2f}' if v is not None else '  -  '
    disp = (r['real'] or r['name'])
    ud = f'{r["user_d"]:.1f}' if r['user_d'] is not None else '-'
    print(f'{str(disp)[:24]:<24} {ud:>6} {fmt(p["EZ"]):>6} {fmt(p["HD"]):>6} {fmt(p["IN"]):>6} {fmt(p["AT"]):>6}')

if skipped:
    print()
    print('===== 未能解析 =====')
    for fn, err in skipped:
        print(f'  {fn}: {err}')

# 统计: 外推区间 (>17.5) vs 内推区间 (只统计有定数的谱)
labeled = [r for r in rows if r['user_d'] is not None]
ext = [r for r in labeled if r['user_d'] > 17.5]
intr = [r for r in labeled if r['user_d'] <= 17.5]
print()
if ext:
    mae_ext = sum(abs(r['pred'] - r['user_d']) for r in ext) / len(ext)
    print(f'外推区间 (用户定数>17.5, n={len(ext)}): MAE={mae_ext:.3f}')
    print('  ' + '  '.join(f'{r["name"]}:{r["pred"]:.1f}({r["user_d"]:.1f})' for r in ext))
if intr:
    mae_int = sum(abs(r['pred'] - r['user_d']) for r in intr) / len(intr)
    print(f'内推区间 (用户定数<=17.5, n={len(intr)}): MAE={mae_int:.3f}')
    print('  ' + '  '.join(f'{r["name"]}:{r["pred"]:.1f}({r["user_d"]:.1f})' for r in intr))
