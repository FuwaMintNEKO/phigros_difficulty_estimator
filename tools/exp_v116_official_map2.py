# -*- coding: utf-8 -*-
"""官谱三角对照v2: 完整预测(gb+boost) vs 官方定数 vs 玩家感知
"""
import os, sys, io, pickle, numpy as np, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT
m = pickle.load(open(os.path.join(_ROOT, 'models', '6dim_model_v11_5c.pkl'), 'rb'))
gb, scaler = m['gb'], m['scaler']
FN = m['feature_names']; LV_ORDER = m['lv_order']
P95 = m['p95_vals']; P99 = m['p99_vals']
FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT); CAPS = m.get('caps', {})
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = cache['official']

def predict_full(feats, level):
    lv = level.upper()
    if 'IN_AT' in LV_ORDER and lv in ('IN','AT'): lv = 'IN_AT'
    if lv not in LV_ORDER: lv = LV_ORDER[-1]
    vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(lv)] = 1.0
    x = np.array([[feats.get(n,0) for n in FN] + vec])
    p_gb = float(gb.predict(scaler.transform(x))[0])
    total = 0.0; cd_ = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0); pv = P95.get(fname, 0)
        t = max(pv*0.55, bl*0.5)
        if v <= t: continue
        e = v/t - 1.0
        c = CAPS.get(fname, cd_)
        if c is not None and e > c: e = c
        x_ = co * (e**0.70)
        p99 = max(P99.get(fname,0), bl*0.5)
        if v > p99:
            pe = v/p99 - 1.0
            if c is not None and pe > c: pe = c
            x_ += co*max(0,pe)**0.70*0.5
        total += x_
    return p_gb + total, p_gb, total

player_map = [
    ('Igallta AT', 'Igallta', 'AT', '玩家: 榜首(公认最强)'),
    ('Igallta IN', 'Igallta', 'IN', '玩家: 15.5合理(未超痉挛in)'),
    ('Spasmodic AT 痉挛', 'Spasmodic', 'AT', '玩家16.45: 楼梯+两侧配置棘手'),
    ('Spasmodic IN 痉挛', 'Spasmodic', 'IN', '玩家15.45: 复杂交互'),
    ('volcanic AT 火山', 'volcanic', 'AT', '玩家16.4: 底力五键连打'),
    ('Lyrith迷宫 AT', 'Lyrith', 'AT', '玩家16.1: 四键掌握'),
    ('Chronostasis 时停', 'Chronostasis', 'IN', '玩家: 倒打段, ~16'),
    ('CrossSoul 穿魂', 'CROSSSOUL', 'IN', '玩家: 需多指, 15.9档最强'),
    ('nmr 旧谱', 'NonMelodic', 'IN', '玩家15.88: 配置繁琐反人类'),
    ('Cthugha IN 古神', 'Cthugha', 'IN', '玩家15.85: 底力需求巨大'),
    ('RIPPER', 'RIPPER', 'IN', '玩家15.84: 慢速高密度冗长'),
    ('SIGMA 死个马', 'SIGMA', 'IN', '玩家15.83'),
    ('GOODRAGE 良怒', 'GOODRAGE', 'IN', '玩家15.8: 底力地狱'),
    ('狂喜蘭舞 AT', '狂喜蘭舞', 'AT', '玩家15.75: 中段多押+尾杀'),
    ('望影の方舟', '方舟', 'IN', '玩家15.68: 读谱+交互门槛'),
    ('Aleph0 a0', 'Aleph0', 'IN', '玩家15.6: 变速tap欺骗'),
    ('雪降りメリクリ', '雪降り', 'IN', '玩家15.38: 三纵连压迫感'),
    ('Reimei', 'Reimei', 'IN', '玩家15.1: 基本无难点'),
    ('Sein', 'Sein', 'IN', '玩家15.1: 14级曲表现突出'),
    ('DistortedFate DF', 'DistortedFate', 'AT', '玩家: 前半简单显虚高'),
    ('DESTRUCTION321', 'DESTRUCTION321', 'AT', '玩家: 略高, 实际15.8'),
    ('祈', '祈', 'AT', '玩家: 后段爆发极难'),
    ('ChronosCollapse 李斯特', 'ChronosCollapse', 'IN', '玩家: 纵连交互叠键全程'),
    ('Shadow 影子', 'Shadow', 'AT', '玩家: 读谱难多指'),
    ('玩具狂奏曲 AT', '玩具狂奏曲', 'AT', '玩家: 16级排行常客'),
    ('BTTB', 'BreakThrough', 'IN', '玩家: 虚低! 实际15.3'),
]
print(f'{"谱面":<24}{"难度":<5}{"官方":>6}{"预测":>6}{"gb":>6}{"boost":>6} | 玩家感知')
for name, kw, lv, note in player_map:
    hits = [r for r in official if kw in r['name'] and r['level'] == lv]
    if not hits: hits = [r for r in official if kw in r['name']]
    if not hits:
        print(f'{name:<24} 未找到'); continue
    r = hits[0]
    p, g_, b_ = predict_full(r['feats'], r['level'])
    print(f'{name:<24}{r["level"]:<5}{r["diff"]:>6.2f}{p:>6.2f}{g_:>6.2f}{b_:>6.2f} | {note}')
print('DONE')
