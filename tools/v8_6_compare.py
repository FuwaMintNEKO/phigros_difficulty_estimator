"""
v8.5 vs v8.6 对比评估
=====================
分层抽样 hold-out 测试 (13-17 定数范围)
对比两个模型的 MAE, 偏差分布, 极端谱预测
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os, pickle, numpy as np, math, random
sys.path.insert(0, '.')
from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from sklearn.metrics import mean_absolute_error
import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

random.seed(42); np.random.seed(42)

print('='*70)
print('  v8.5 vs v8.6 对比评估')
print('='*70)

# ===== 加载数据 =====
CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
diff_map = load_difficulty_tsv(TSV)
chart_files = find_chart_files(CHART_DIR)

all_charts = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in diff_map: continue
    for lv in ['IN','AT']:
        if lv not in info.get('levels',{}): continue
        if lv not in diff_map[sid]: continue
        try:
            cd = load_chart_json(info['levels'][lv]); f = extract_features(cd)
            if f: f['_difficulty'] = diff_map[sid][lv]; f['_name'] = fn[:30]; all_charts.append(f)
        except: pass

exclude = ['chartnekockLK','snow dance','Snow']
all_charts = [f for f in all_charts if not any(p.lower() in f['_name'].lower() for p in exclude)]
print(f'官谱总数: {len(all_charts)}')

diffs_all = np.array([f['_difficulty'] for f in all_charts])

# 分层抽样 hold-out
hold_out = np.zeros(len(all_charts), dtype=bool)
bins = np.digitize(diffs_all, bins=[13,14,15,16,17])
for b in range(1,6):
    idx = np.where(bins == b)[0]
    if len(idx) < 3: continue
    n_hold = max(1, int(len(idx) * 0.25))
    chosen = np.random.choice(idx, size=n_hold, replace=False)
    hold_out[chosen] = True

test_charts = [all_charts[i] for i in range(len(all_charts)) if hold_out[i]]
y_test = diffs_all[hold_out]
print(f'Hold-out: {len(test_charts)} charts (13-17 range)')

# ===== 加载两个模型 =====
with open('models/6dim_model_v8_5.pkl', 'rb') as f: m85 = pickle.load(f)
with open('models/6dim_model_v8_6.pkl', 'rb') as f: m86 = pickle.load(f)

def compute_excess(feats, fname, bl, P95, P99):
    val = feats.get(fname, 0)
    pv = P95.get(fname, 0)
    thresh = max(pv * 0.55, bl * 0.5)
    if val <= thresh: return 0.0
    excess = (val / thresh - 1.0) ** 0.70
    if val > max(P99.get(fname, 0), bl * 0.5):
        pe = (val / max(P99.get(fname, 0), bl * 0.5) - 1.0) ** 0.70
        excess += 0.5 * max(0, pe) ** 0.70
    return excess

def compute_raw_boost(feats, FLAT, P95, P99):
    raw = 0.0
    for fname, bl, co in FLAT:
        raw += co * compute_excess(feats, fname, bl, P95, P99)
    return raw

def _dc(r, DC):
    if r <= DC['knee']: return r
    return DC['knee'] + (r - DC['knee']) ** DC['power']

def adjust_boost_smooth(boost, gb, target=0.32, power=0.75, thresh=0.22, steepness=25):
    if boost < 2.0: return boost
    ratio = boost / gb if gb > 0 else 0
    expected = target * gb
    if expected <= 0 or boost <= 0: return boost
    adj = expected * ((boost / expected) ** power)
    w = 1 / (1 + math.exp(-steepness * (ratio - thresh)))
    return (1 - w) * boost + w * adj

def predict(feats, model):
    gb = model['gb']; sc = model['scaler']; FN = model['feature_names']
    FLAT = model['FLAT_FEATURES']; P95 = model['p95_vals']; P99 = model['p99_vals']
    DC = model['dynamic_cap']
    sig = model.get('sigmoid_params', {'target': 0.32, 'power': 0.75, 'thresh': 0.22})
    
    X = np.array([[feats.get(n, 0) for n in FN]])
    gb_raw = gb.predict(sc.transform(X))[0]
    boost = _dc(compute_raw_boost(feats, FLAT, P95, P99), DC)
    adj_boost = adjust_boost_smooth(boost, gb_raw, sig['target'], sig['power'], sig['thresh'])
    return gb_raw + adj_boost

preds_85 = np.array([predict(f, m85) for f in test_charts])
preds_86 = np.array([predict(f, m86) for f in test_charts])

mae_85 = mean_absolute_error(y_test, preds_85)
mae_86 = mean_absolute_error(y_test, preds_86)

print(f'\nv8.5 Hold-out MAE: {mae_85:.4f} ({len(m85["FLAT_FEATURES"])} features)')
print(f'v8.6 Hold-out MAE: {mae_86:.4f} ({len(m86["FLAT_FEATURES"])} features)')

if mae_86 < mae_85:
    print(f'  >> IMPROVEMENT: -{mae_85 - mae_86:.4f} ({((mae_85 - mae_86)/mae_85)*100:.1f}%)')
else:
    print(f'  >> REGRESSION: +{mae_86 - mae_85:.4f}')

# 详细偏差分析
print(f'\n===== 偏差分布 =====')
errors_85 = preds_85 - y_test
errors_86 = preds_86 - y_test

print(f'{"":20s} {"v8.5":>10s} {"v8.6":>10s}')
print(f'{"Mean Error":20s} {np.mean(errors_85):>10.4f} {np.mean(errors_86):>10.4f}')
print(f'{"Std Error":20s} {np.std(errors_85):>10.4f} {np.std(errors_86):>10.4f}')
print(f'{"Max Over":20s} {np.max(errors_85):>10.4f} {np.max(errors_86):>10.4f}')
print(f'{"Max Under":20s} {np.min(errors_85):>10.4f} {np.min(errors_86):>10.4f}')

# 按定数区间统计
print(f'\n===== 按定数区间 MAE =====')
for lo, hi in [(13,14),(14,15),(15,16),(16,17),(17,20)]:
    mask = (y_test >= lo) & (y_test < hi)
    if mask.sum() == 0: continue
    m85i = mean_absolute_error(y_test[mask], preds_85[mask])
    m86i = mean_absolute_error(y_test[mask], preds_86[mask])
    d = m86i - m85i
    sign = '+' if d > 0 else ''
    print(f'  [{lo},{hi}) n={mask.sum():2d}  v8.5={m85i:.4f}  v8.6={m86i:.4f}  Δ={sign}{d:.4f}')

# 极端谱分析
print(f'\n===== 极端谱预测 (偏差 > 0.5) =====')
print(f'{"Name":<30s} {"True":>6s} {"v8.5":>6s} {"v8.6":>6s} {"Δ85":>7s} {"Δ86":>7s}')
for i in range(len(test_charts)):
    e85 = abs(errors_85[i])
    e86 = abs(errors_86[i])
    if max(e85, e86) > 0.4:
        name = test_charts[i]['_name'][:28]
        print(f'{name:<30s} {y_test[i]:>6.1f} {preds_85[i]:>6.2f} {preds_86[i]:>6.2f} {errors_85[i]:>+7.2f} {errors_86[i]:>+7.2f}')

# 全量训练集 MAE
print(f'\n===== 全量训练集 MAE (参照) =====')
all_preds_85 = np.array([predict(f, m85) for f in all_charts])
all_preds_86 = np.array([predict(f, m86) for f in all_charts])
print(f'v8.5 全量 MAE: {mean_absolute_error(diffs_all, all_preds_85):.4f}')
print(f'v8.6 全量 MAE: {mean_absolute_error(diffs_all, all_preds_86):.4f}')

print(f'\n===== 完成 =====')