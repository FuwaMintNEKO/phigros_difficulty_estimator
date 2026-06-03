"""v5.2 测试谱评估脚本 — 使用 app.py 的 boost 逻辑，自动同步"""
import os, sys, pickle, numpy as np
sys.path.insert(0, '.')
from feature_extractor import extract_features
from unified_parser import load_chart

# ====== 从 app.py 导入最新的 boost 逻辑 ======
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', '5dim_model_v5_3.pkl')
with open(MODEL_PATH, 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']

# 从 pickle 加载 FLAT_FEATURES（和训练脚本同步）
FLAT_FEATURES = m.get('FLAT_FEATURES', [])
DC = m.get('dynamic_cap', {'knee': 2.5, 'power': 0.9})

def _dynamic_cap(raw):
    KNEE = DC['knee']; POWER = DC['power']
    if raw <= KNEE: return raw
    excess = raw - KNEE
    return KNEE + excess ** POWER

def compute_boost(feats):
    contribs = []
    for fname, bl, co in FLAT_FEATURES:
        v = feats.get(fname, 0)
        pv = P95.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t: continue
        excess = v / t - 1.0
        c = co * (excess ** 0.55)
        p99 = P99.get(fname, 0)
        if v > max(p99, bl * 0.5):
            c += co * max(0, v / max(p99, bl * 0.5) - 1.0) ** 0.55 * 0.5
        contribs.append((fname, c, v, t, co))
    total = sum(c for _, c, _, _, _ in contribs)
    total = _dynamic_cap(total)
    return total, contribs

# ====== 测试谱面列表 ======
TD = r'D:\迅雷下载\Phigros_Resource-master\Phigros_Resource-master\chart\test_datas'
DL = r'C:\Users\NaNK\Downloads'

charts = [
    ("DA'AT",                   TD, '2155734445357448.json',        18.2),
    ("WakingShadows",           TD, '93562988.json',                18.1),
    ("Chart_SP #13",            TD, 'Chart_SP #1347(1).json',       17.6),
    ("105秒伝說",               TD, 'Sigma (Haocore Mix) ~ 105秒の伝說 ~.json', 16.1),
    ("LiFE Garden(1.05x)",      TD, '6923526264684294.json',        17.9),
    ("Far Eastern Flavor",      DL, '61901444.json',                17.5),
    ("密码的周一",              DL, '0582581966828779.json',        17.4),
    ("People people",           DL, '1770391855.json',              None),
    ("Galaxy Collapse",         DL, '7009367902368871.json',        None),
    ("Apollo",                  DL, 'Apollo(18.0).json',            18.0),
    ("Xaleid◆scopiX",           DL, 'Xaleid◆scopiX(18.2)(1).json', 18.2),
    ("silly-willy-nilly",       DL, 'silly-willy-nilly(17.9)(1).json', 17.9),
    ("おぎゃりないざー",        DL, 'おぎゃりないざー(16.5~16.6).json', 16.55),
    ("恋ひ恋ふ縁",              DL, '恋ひ恋ふ縁(16.8)(1).json',    16.8),
    ("朧月",                    DL, '朧月(18.4)(1).json',          18.4),
    ("ニャンだふる♡サマー!!",   DL, 'ニャンだふる♡サマー!!(15.8).json', 15.8),
    ("Breakcore革命前夜",       DL, 'Breakcore革命前夜(16.3~16.5).json', 16.4),
    ("Cheerio!",                DL, 'Cheerio!(17.1).json',          17.1),
    ("Lemegeton",               DL, 'Lemegeton -little key of solomon-(16.6).json', 16.6),
    ("Submerged City",          DL, 'Submerged City(18.0).json',    18.0),
    ("Rrharil AT",              r'D:\迅雷下载\Phigros_Resource-master\Phigros_Resource-master\chart\Rrharil.TeamGrimoire.0', 'AT.json', 17.6),
    ("QZKagoRequiem AT",        r'D:\迅雷下载\Phigros_Resource-master\Phigros_Resource-master\chart\QZKagoRequiem.tpazolite.0', 'AT.json', 17.4),
]

# ====== 运行 ======
print(f'{"谱面":<24} {"预期":>6} {"GB":>7} {"Boost":>7} {"预测":>7} {"误差":>8}')
print('-'*65)
errs = []
for name, basedir, fname, exp in charts:
    fp = os.path.join(basedir, fname)
    try:
        cd = load_chart(fp)
        fe = extract_features(cd)
        if fe is None:
            print(f'{name:<24} {"提取失败":>10}')
            continue
        x = np.array([[fe.get(n, 0) for n in FN]])
        xs = scaler.transform(x)
        gv = float(gb.predict(xs)[0])
        bv, contribs = compute_boost(fe)
        p = gv + bv

        es = f'{exp:.2f}' if exp else '  ?'
        err = f'{p-exp:+.3f}' if exp else '  -'
        mark = '' if exp is None or abs(p-exp) < 0.31 else '⚠️' if abs(p-exp) < 0.5 else '🔴'
        print(f'{name:<24} {es:>6} {gv:>7.3f} {bv:>7.3f} {p:>7.3f} {err:>8} {mark}')
        if exp:
            errs.append(abs(p-exp))
    except Exception as e:
        print(f'{name:<24} ERROR: {e}')

if errs:
    good = sum(1 for e in errs if e < 0.31)
    print(f'\n有预期 {len(errs)} 张: 平均={np.mean(errs):.3f}  最大={np.max(errs):.3f}  <0.3={good}/{len(errs)}')
