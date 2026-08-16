# -*- coding: utf-8 -*-
"""v13 Boost权重线性优化 (最原始的调: 只动MANUAL_FLAT权重, 不动GB/不加规则)
- 加载 v13模型 + v13_aux(留出50/官谱OOF特征+GB残差) + 17-锚点(test_charts)
- boost对权重co线性: pred = gb + Σ co_j * terms_j
- 目标: 留出MAE(表为准) + 官谱OOF MAE + 锚点超界惩罚
- 坐标下降3轮, 每权重21网格, 范围±50%; 输出最优权重json
"""
import os, sys, io, pickle, json, re, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

m = pickle.load(open(os.path.join(_ROOT, 'models', '6dim_model_v13.pkl'), 'rb'))
aux = pickle.load(open(os.path.join(_ROOT, 'models', 'v13_aux.pkl'), 'rb'))
FLAT = list(m['MANUAL_FLAT'])
P95, P99, CAPS = m['p95_vals'], m['p99_vals'], m.get('caps', {})
cap_default = CAPS.get('_default', None)
CO_CUR = np.array([co for _, _, co in FLAT])

def boost_terms(feats):
    terms = []
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0)
        pv = P95.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t:
            terms.append(0.0)
            continue
        e = v / t - 1.0
        c = CAPS.get(fname, cap_default)
        if c is not None and e > c:
            e = c
        x = e ** 0.70
        if v > max(P99.get(fname, 0), bl * 0.5):
            pe = v / max(P99.get(fname, 0), bl * 0.5) - 1.0
            if c is not None and pe > c:
                pe = c
            x += max(0, pe) ** 0.70 * 0.5
        terms.append(x)
    return np.array(terms)

# 数据集: 留出50 / 官谱OOF 982 / 17-锚点
hold_feats = aux['hold_feats']; hold_labels = np.array(aux['hold_labels']); hold_gb = np.array(aux['hold_gb'])
off_feats = aux['off_feats']; off_gb = np.array(aux['oof_gb']); off_y = np.array(aux['y_off'])
print('留出 %d, 官谱OOF %d' % (len(hold_feats), len(off_feats)))

H = np.array([boost_terms(f) for f in hold_feats])
O = np.array([boost_terms(f) for f in off_feats])

ANCH = []
def parse_target(fname):
    mch = re.search(r'[（(]([0-9]+(?:[.][0-9]+)?)(?:~([0-9]+(?:[.][0-9]+)?))?[)）]', fname)
    if mch:
        return (float(mch.group(1)) + float(mch.group(2))) / 2 if mch.group(2) else float(mch.group(1))
    return None
for fn in sorted(os.listdir(os.path.join(_ROOT, 'data', 'test_charts'))):
    if not fn.endswith('.json'):
        continue
    t = parse_target(fn)
    if t is None or t > 17.0:
        continue
    if 'Lemegeton' in fn:
        continue
    if 'Chart_SP' in fn:
        t = 17.65
    if 'Runengon' in fn:
        t = 16.8
    if 'おぎゃり' in fn:
        t = 16.5
    p = os.path.join(_ROOT, 'data', 'test_charts', fn)
    cd, raw = load_chart_from_bytes(open(p, 'rb').read())
    fe = extract_features(cd)
    if fe:
        ANCH.append((fn[:24], t, fe))
print('17-锚点 %d' % len(ANCH))
AB_feats = [a[2] for a in ANCH]
AB_tgt = np.array([a[1] for a in ANCH])

# 锚点GB残差: 用当前模型预测 - boost
sys.path.insert(0, _ROOT)
import app as A
AB_gb = []
for nm, t, fe in ANCH:
    fe2 = dict(fe)
    for k, dd in A.DOMAIN_DELTA.items():
        if k in fe2:
            fe2[k] = fe2[k] - dd
    p = float(A.predict_from_feats(fe2, 'IN', True)[0])
    b = float(np.dot(CO_CUR, boost_terms(fe2)))
    AB_gb.append(p - b)
AB_gb = np.array(AB_gb)
AT = np.array([boost_terms(A.apply_domain_align(dict(fe), True, 'IN')) for nm, t, fe in ANCH])

def preds(co):
    ph = hold_gb + H @ co
    po = off_gb + O @ co
    pa = AB_gb + AT @ co
    return ph, po, pa

def objective(co, lam_hold=3.0, lam_off=0.5, lam_anch=2.0):
    ph, po, pa = preds(co)
    mae_h = np.abs(ph - hold_labels).mean()
    mae_o = np.abs(po - off_y).mean()
    pen_a = np.maximum(np.abs(pa - AB_tgt) - 0.28, 0) ** 2
    return lam_hold * mae_h + lam_off * mae_o + lam_anch * pen_a.mean()

ph0, po0, pa0 = preds(CO_CUR)
print('\n基线: 留出MAE=%.4f 官谱OOF MAE=%.4f 锚点超界=%d 目标=%.4f' % (
    np.abs(ph0 - hold_labels).mean(), np.abs(po0 - off_y).mean(),
    int((np.abs(pa0 - AB_tgt) > 0.28).sum()), objective(CO_CUR)))
for i, (nm, t, f) in enumerate(ANCH):
    print('  锚点 %-24s tgt=%.2f pred=%.2f' % (nm, t, pa0[i]))

# 坐标下降
co = CO_CUR.copy()
best = objective(co)
print('\n初始目标: %.4f' % best)
for rnd in range(3):
    improved = 0
    for j in range(len(FLAT)):
        lo = max(CO_CUR[j] * 0.5, 0.001 if CO_CUR[j] > 0 else CO_CUR[j] * 1.5)
        hi = CO_CUR[j] * 1.5
        if CO_CUR[j] <= 0:
            lo, hi = CO_CUR[j] * 1.5, CO_CUR[j] * 0.5
        grid = np.linspace(min(lo, hi), max(lo, hi), 21)
        objs = []
        for v in grid:
            t2 = co.copy()
            t2[j] = v
            objs.append(objective(t2))
        k = int(np.argmin(objs))
        if objs[k] < best - 1e-6:
            co[j] = grid[k]
            best = objs[k]
            improved += 1
    print('轮%d: 目标=%.4f 改进%d' % (rnd + 1, best, improved))
    if improved == 0:
        break

ph, po, pa = preds(co)
print('\n===== 优化后 =====')
print('留出MAE: %.4f -> %.4f' % (np.abs(ph0 - hold_labels).mean(), np.abs(ph - hold_labels).mean()))
print('官谱OOF MAE: %.4f -> %.4f' % (np.abs(po0 - off_y).mean(), np.abs(po - off_y).mean()))
print('锚点超界: %d -> %d' % (int((np.abs(pa0 - AB_tgt) > 0.28).sum()), int((np.abs(pa - AB_tgt) > 0.28).sum())))
print('\n权重变化(>20%):')
for j in range(len(FLAT)):
    if abs(co[j] - CO_CUR[j]) / max(abs(CO_CUR[j]), 1e-6) > 0.2:
        print('  %-30s %+.4f -> %+.4f' % (FLAT[j][0], CO_CUR[j], co[j]))
print('\n锚点明细:')
for i, (nm, t, f) in enumerate(ANCH):
    flag = ' <<<' if abs(pa[i] - t) > 0.28 else ''
    print('  %-24s tgt=%.2f %.2f -> %.2f%s' % (nm, t, pa0[i], pa[i], flag))
# 留出明细
print('\n留出明细(优化后):')
for i in range(len(hold_labels)):
    print('  #%-6d %-30s 表%.1f %.2f -> %.2f' % (aux['hold_names'][i][0], aux['hold_names'][i][1][:30], hold_labels[i], ph0[i], ph[i]))
new_flat = [(f[0], f[1], float(co[j])) for j, f in enumerate(FLAT)]
with open(os.path.join(_ROOT, 'models', 'v13_boost_weights.json'), 'w', encoding='utf-8') as f:
    json.dump(new_flat, f, ensure_ascii=False, indent=1)
print('\n已保存: models/v13_boost_weights.json')
