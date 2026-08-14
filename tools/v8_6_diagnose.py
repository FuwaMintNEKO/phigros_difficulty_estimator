"""
v8.6 诊断: 检查 GB 输出 vs Boost 输出
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os, pickle, numpy as np, math
sys.path.insert(0, '.')
from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json

with open('models/6dim_model_v8_6.pkl', 'rb') as f: m = pickle.load(f)

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

gb = m['gb']; sc = m['scaler']; FN = m['feature_names']
FLAT = m['FLAT_FEATURES']; P95 = m['p95_vals']; P99 = m['p99_vals']
DC = m['dynamic_cap']

def compute_excess(feats, fname, bl):
    val = feats.get(fname, 0); pv = P95.get(fname, 0)
    thresh = max(pv * 0.55, bl * 0.5)
    if val <= thresh: return 0.0
    excess = (val / thresh - 1.0) ** 0.70
    if val > max(P99.get(fname, 0), bl * 0.5):
        pe = (val / max(P99.get(fname, 0), bl * 0.5) - 1.0) ** 0.70
        excess += 0.5 * max(0, pe) ** 0.70
    return excess

def compute_raw_boost(feats):
    raw = 0.0
    for fname, bl, co in FLAT:
        raw += co * compute_excess(feats, fname, bl)
    return raw

def _dc(r):
    if r <= DC['knee']: return r
    return DC['knee'] + (r - DC['knee']) ** DC['power']

def adjust_boost_smooth(boost, gb_val):
    if boost < 2.0: return boost
    ratio = boost / gb_val if gb_val > 0 else 0
    expected = 0.32 * gb_val
    if expected <= 0 or boost <= 0: return boost
    adj = expected * ((boost / expected) ** 0.75)
    w = 1 / (1 + math.exp(-25 * (ratio - 0.22)))
    return (1 - w) * boost + w * adj

# 选一些有代表性的谱面
sample_names = [
    'BANGINGSTRIKE','DESTRUCTION321','Nhelv','ReEndofaDream',
    'FlutterEcho','狂喜蘭舞','游园地','JourneywithYou',
    'Cuvism','csqn','DataErr0r','HorizonBlue',
    'Lyrith','KIZUNAResolution','slips','SATELLITE',
    '乱舞','DerRichter','INFiNiTEENERZY',
]

print(f'{"Name":<30s} {"True":>6s} {"GB":>7s} {"Boost":>7s} {"AdjBst":>7s} {"Pred":>7s} {"Err":>7s}')
print('-'*75)

for name_pat in sample_names:
    for c in all_charts:
        if name_pat.lower() in c['_name'].lower():
            feats = c
            X = np.array([[feats.get(n, 0) for n in FN]])
            gb_raw = gb.predict(sc.transform(X))[0]
            boost_raw = _dc(compute_raw_boost(feats))
            adj_boost = adjust_boost_smooth(boost_raw, gb_raw)
            pred = gb_raw + adj_boost
            true_val = c['_difficulty']
            name = c['_name'][:28]
            print(f'{name:<30s} {true_val:>6.1f} {gb_raw:>7.2f} {boost_raw:>7.2f} {adj_boost:>7.2f} {pred:>7.2f} {pred-true_val:>+7.2f}')
            break

# 统计
print(f'\n===== 全量统计 =====')
all_gb = []; all_boost = []; all_pred = []; all_true = []
for c in all_charts:
    X = np.array([[c.get(n, 0) for n in FN]])
    gb_raw = gb.predict(sc.transform(X))[0]
    boost_raw = _dc(compute_raw_boost(c))
    adj_boost = adjust_boost_smooth(boost_raw, gb_raw)
    all_gb.append(gb_raw)
    all_boost.append(boost_raw)
    all_pred.append(gb_raw + adj_boost)
    all_true.append(c['_difficulty'])

all_gb = np.array(all_gb); all_boost = np.array(all_boost)
all_pred = np.array(all_pred); all_true = np.array(all_true)

print(f'GB range:     [{all_gb.min():.2f}, {all_gb.max():.2f}]')
print(f'Boost range:  [{all_boost.min():.2f}, {all_boost.max():.2f}]')
print(f'Pred range:   [{all_pred.min():.2f}, {all_pred.max():.2f}]')
print(f'True range:   [{all_true.min():.2f}, {all_true.max():.2f}]')
print(f'GB/Boost ratio: mean={np.mean(all_gb/all_boost):.2f} (excluding boost=0)')
print(f'MAE: {np.mean(np.abs(all_pred - all_true)):.4f}')

# 检查 P95/P99 是否来自训练数据
print(f'\n===== P95/P99 检查 =====')
for n, b, c in FLAT[:5]:
    vals = [f.get(n, 0) for f in all_charts]
    print(f'{n}: P95={P95[n]:.2f} (computed={np.percentile(vals, 95):.2f}) P99={P99[n]:.2f} (computed={np.percentile(vals, 99):.2f})')