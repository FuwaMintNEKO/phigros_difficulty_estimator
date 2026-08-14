# -*- coding: utf-8 -*-
"""官谱三角对照: 玩家感知(文章) vs 官方定数 vs v11.5c预测 + 关键特征
"""
import os, sys, io, pickle, numpy as np, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
m = pickle.load(open(os.path.join(_ROOT, 'models', '6dim_model_v11_5c.pkl'), 'rb'))
gb, scaler = m['gb'], m['scaler']
FN = m['feature_names']; LV_ORDER = m['lv_order']
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = cache['official']

# 玩家文章提到的谱 (文件夹名关键字)
player_map = [
    ('Igallta AT', 'Igallta', 'AT', '官方最高难度之一, 玩家评榜首'),
    ('Igallta IN', 'Igallta', 'IN', '玩家: 实际未超痉挛in, 15.5合理'),
    ('Spasmodic AT (痉挛)', 'Spasmodic', 'AT', '玩家16.45: 楼梯段+两侧配置棘手'),
    ('Spasmodic IN (痉挛)', 'Spasmodic', 'IN', '玩家15.45: 复杂交互协调'),
    ('volcanic AT (火山)', 'volcanic', 'AT', '玩家16.4: 底力, 五键连打'),
    ('Lyrith迷宫 AT', 'Lyrith', 'AT', '玩家16.1: 四键掌握'),
    ('Chronostasis (时停)', 'Chronostasis', 'IN', '玩家: 倒打段, 不转板16'),
    ('CrossSoul (穿魂)', 'CROSSSOUL', 'IN', '玩家: 15.9实为16.1-16.2, 需多指'),
    ('NonMelodicRagez (nmr)', 'NonMelodic', 'IN', '玩家15.88: 配置繁琐反人类, 读谱压力'),
    ('Cthugha IN (古神)', 'Cthugha', 'IN', '玩家15.85: 底力需求巨大'),
    ('RIPPER', 'RIPPER', 'IN', '玩家15.84: 慢速高密度, 冗长'),
    ('SIGMA (死个马)', 'SIGMA', 'IN', '玩家15.83'),
    ('GOODRAGE (良怒)', 'GOODRAGE', 'IN', '玩家15.8: 底力地狱'),
    ('狂喜蘭舞 AT', '狂喜蘭舞', 'AT', '玩家15.75: 中段多押+结尾杀'),
    ('望影の方舟 (方舟)', '方舟', 'IN', '玩家15.68: 读谱+交互门槛'),
    ('Aleph0 (a0)', 'Aleph0', 'IN', '玩家15.6: 变速难点, tap欺骗性'),
    ('雪降りメリクリ (雪降)', '雪降り', 'IN', '玩家15.38: 三纵连压迫感'),
    ('Reimei', 'Reimei', 'IN', '玩家15.1: 基本无难点'),
    ('Sein', 'Sein', 'IN', '玩家15.1: 14级曲表现突出'),
    ('DistortedFate (DF)', 'DistortedFate', 'AT', '玩家: 前半简单, 虚高'),
    ('DESTRUCTION321', 'DESTRUCTION321', 'AT', '玩家: 定数略高, 实际15.8'),
    ('祈', '祈', 'AT', '玩家: 后段爆发极难'),
    ('ChronosCollapse (李斯特)', 'ChronosCollapse', 'IN', '玩家: 纵连交互叠键全程'),
    ('Shadow (影子)', 'Shadow', 'AT', '玩家: 读谱难多指, 十字架段'),
    ('玩具狂奏曲', '玩具狂奏曲', 'AT', '玩家: 16级排行常客'),
    ('BreakThrough (BTTB)', 'BreakThrough', 'IN', '玩家: 虚低! 实际15.3 (14级谱)'),
]

def predict(feats, level):
    lv = level.upper()
    if 'IN_AT' in LV_ORDER and lv in ('IN','AT'): lv = 'IN_AT'
    if lv not in LV_ORDER: lv = LV_ORDER[-1]
    vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(lv)] = 1.0
    x = np.array([[feats.get(n,0) for n in FN] + vec])
    return float(gb.predict(scaler.transform(x))[0])

print(f'{"谱面":<26}{"难度":<6}{"官方":>6}{"预测":>6} | 玩家感知')
for name, kw, lv, note in player_map:
    hits = [r for r in official if kw in r['name'] and r['level'] == lv]
    if not hits:
        # 宽松匹配
        hits = [r for r in official if kw in r['name']]
    if not hits:
        print(f'{name:<26} 未找到 ({kw}/{lv})')
        continue
    r = hits[0]
    p = predict(r['feats'], r['level'])
    print(f'{name:<26}{r["level"]:<6}{r["diff"]:>6.2f}{p:>6.2f} | {note}')
print('DONE')
