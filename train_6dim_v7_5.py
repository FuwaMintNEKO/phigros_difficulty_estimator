"""
v7.5 — 倍速增强训练：GB+boost都天然响应速度变化
  → slider = 改文件, 完全一致, 无需任何speed参数
"""
import sys, json, os, pickle, copy, numpy as np, math
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, '.')
from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print('=' * 70)
print('  v7.5 — 倍速增强训练')
print('=' * 70)

# ── 加载基础参数 ──
with open('models/6dim_model_v7_2.pkl', 'rb') as f:
    v7m = pickle.load(f)
P95 = v7m['p95_vals']; P99 = v7m['p99_vals']; FN_GB = v7m['feature_names']
with open('models/6dim_model_v7_3.pkl', 'rb') as f:
    v73m = pickle.load(f)
FLAT_ORIG = v73m['FLAT_FEATURES']
feat_boost_names = [f[0] for f in FLAT_ORIG]; n_boost = len(feat_boost_names)
DC = {'knee': 1.0, 'power': 0.90}

SPEED_CURVE = {0.5:-3.5, 0.6:-2.5, 0.7:-1.8, 0.8:-1.0, 0.85:-0.5, 0.9:-0.2,
               1.0:0.0, 1.1:0.3, 1.15:0.6, 1.2:1.0, 1.3:1.5, 1.4:2.0,
               1.5:2.5, 1.6:3.0, 1.7:3.5, 1.8:4.0, 2.0:5.0}
SPEEDS = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]

def apply_speed(data, s):
    d = copy.deepcopy(data)
    for jl in d.get('judgeLineList', []):
        if 'bpm' in jl: jl['bpm'] = jl['bpm'] * s
    for e in d.get('BPMList', []):
        if 'bpm' in e: e['bpm'] = e['bpm'] * s
    return d

def compute_excess(feats, fname, bl):
    v = feats.get(fname, 0)
    pv = P95.get(fname, 0); t = max(pv * 0.55, bl * 0.5)
    if v <= t: return 0.0
    e = (v / t - 1.0) ** 0.70
    if v > max(P99.get(fname, 0), bl * 0.5):
        pe = (v / max(P99.get(fname, 0), bl * 0.5) - 1.0)
        e += 0.5 * max(0, pe) ** 0.70
    return e

def compute_raw_boost(feats, co_arr):
    return sum(co_arr[j] * compute_excess(feats, feat_boost_names[j], FLAT_ORIG[j][1]) for j in range(n_boost))

def cap_boost(raw):
    return raw if raw <= DC['knee'] else DC['knee'] + (raw - DC['knee']) ** DC['power']

# ── 加载IN/AT官谱 ──
CHART_DIR = r'D:\迅雷下载\Phigros_Resource-master\Phigros_Resource-master\chart'
DIFFICULTY_TSV = r'D:\迅雷下载\Phigros_Resource-master\Phigros_Resource-master\info\difficulty.tsv'
song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)

all_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    for lv in ['IN', 'AT']:
        if lv in info['levels'] and lv in diffs:
            all_items.append({'folder': fn, 'filepath': info['levels'][lv], 'difficulty': diffs[lv], 'level': lv})

print(f'IN/AT原始谱: {len(all_items)}')

# ── 生成增强数据 ──
feats_list, labels, names = [], [], []
for i, item in enumerate(all_items):
    if (i+1) % 100 == 0: print(f'  加载 {i+1}/{len(all_items)}...')
    try:
        base_data = load_chart_json(item['filepath'])
        base_feats = extract_features(base_data)
        if not base_feats: continue
        feats_list.append(base_feats); labels.append(item['difficulty'])
        names.append(f'{i}/{item["level"]}')  # 用原始索引追踪
        for s in SPEEDS:
            if s == 1.0: continue
            fd = extract_features(apply_speed(base_data, s))
            if fd:
                feats_list.append(fd)
                labels.append(item['difficulty'] + SPEED_CURVE.get(s, (s-1)*4))
                names.append(f'{i}/{item["level"]}_{s}x')
    except: pass

n_all = len(feats_list); labels = np.array(labels)
print(f'增强后总谱: {n_all}, 难度 {labels.min():.1f}~{labels.max():.1f}')

# ── 特征矩阵 ──
X_gb = np.array([[f.get(k, 0) for k in FN_GB] for f in feats_list])
X_excess = np.zeros((n_all, n_boost))
for i in range(n_all):
    for j in range(n_boost):
        X_excess[i, j] = compute_excess(feats_list[i], feat_boost_names[j], FLAT_ORIG[j][1])

# ── 训练/测试划分: 按原始索引分裂，同一谱的速度变体在同一侧 ──
orig_ids = np.array([int(n.split('/')[0]) for n in names])
unique_orig = np.unique(orig_ids)
np.random.seed(42); np.random.shuffle(unique_orig)
n_test = max(5, len(unique_orig) // 7)
test_orig = set(unique_orig[:n_test])
test_mask = np.array([orig_ids[i] in test_orig for i in range(n_all)])
train_mask = ~test_mask
print(f'  训练: {train_mask.sum()} ({len(unique_orig)-n_test}个原始谱), 测试: {test_mask.sum()} ({n_test}个原始谱)')

# ── 迭代优化 ──
co_current = np.array([f[2] for f in FLAT_ORIG])
for it in range(2):
    print(f'\n--- 迭代 {it+1}/2 ---')
    b_all = np.array([cap_boost(compute_raw_boost(feats_list[i], co_current)) for i in range(n_all)])
    X_tr, y_tr, b_tr = X_gb[train_mask], labels[train_mask], b_all[train_mask]
    X_te, y_te, b_te = X_gb[test_mask], labels[test_mask], b_all[test_mask]

    sc = StandardScaler()
    gb_m = GradientBoostingRegressor(n_estimators=300, max_depth=5, min_samples_leaf=5,
                                      learning_rate=0.08, subsample=0.8, random_state=42)
    gb_m.fit(sc.fit_transform(X_tr), y_tr - b_tr)
    p_te = gb_m.predict(sc.transform(X_te)) + b_te
    print(f'  GB: R²={r2_score(y_te, p_te):.4f} MAE={mean_absolute_error(y_te, p_te):.4f}')

    sc_a = StandardScaler(); gb_full = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, min_samples_leaf=5, learning_rate=0.08, subsample=0.8, random_state=42)
    gb_full.fit(sc_a.fit_transform(X_gb), labels - b_all)
    y_res = labels - gb_full.predict(sc_a.transform(X_gb))

    best_a = 1.0; best_cv = float('inf')
    for a in [0.01, 0.1, 1.0, 10.0, 50.0]:
        r = Ridge(alpha=a, fit_intercept=False, positive=True)
        s = -cross_val_score(r, X_excess, y_res, cv=3, scoring='neg_mean_absolute_error')
        if s.mean() < best_cv: best_cv = s.mean(); best_a = a
    ridge = Ridge(alpha=best_a, fit_intercept=False, positive=True)
    ridge.fit(X_excess, y_res)
    print(f'  Ridge α={best_a:.3f}')
    co_current = 0.3 * co_current + 0.7 * ridge.coef_

# ── 最终全量 ──
b_final = np.array([cap_boost(compute_raw_boost(feats_list[i], co_current)) for i in range(n_all)])
sc_final = StandardScaler()
gb_final = GradientBoostingRegressor(n_estimators=700, max_depth=5, min_samples_leaf=5,
                                      learning_rate=0.05, subsample=0.8, random_state=42)
gb_final.fit(sc_final.fit_transform(X_gb), labels - b_final)
print('\n全量训练完成')

# ── 测试谱验证 ──
test_dir = r'C:\Users\NaNK\Downloads'
chart_tests = []
for fn in os.listdir(test_dir):
    if not fn.endswith('.json'): continue
    fp = os.path.join(test_dir, fn)
    if os.path.getsize(fp) < 100: continue
    try:
        import re
        rt = None
        for m in re.finditer(r'\((\d+\.?\d*)\)', fn):
            v = float(m.group(1))
            if 5 <= v <= 20: rt = v; break
        if rt is None: continue
        with open(fp, 'rb') as f: data, _ = __import__('unified_parser').load_chart_from_bytes(f.read())
        chart_tests.append((fn, data, rt))
    except: continue

print(f'\n{"="*70}')
print('倍速一致性验证:')
print('='*70)

def quick_predict(data):
    f = extract_features(data)
    X = sc_final.transform(np.array([[f.get(k, 0) for k in FN_GB]]))
    return float(gb_final.predict(X)[0]) + cap_boost(compute_raw_boost(f, co_current))

for name, fp in [('怪文書', r'C:\Users\NaNK\Downloads\ギザバ怪文書(18.3).json'),
                  ('Apollo', r'C:\Users\NaNK\Downloads\Apollo(17.8).json'),
                  ('恋ひ恋ふ縁', r'C:\Users\NaNK\Downloads\恋ひ恋ふ縁(16.8)(1).json'),
                  ('スタートリップ', r'C:\Users\NaNK\Downloads\スタートリップ(12.2).json')]:
    with open(fp, 'rb') as f: base, _ = __import__('unified_parser').load_chart_from_bytes(f.read())
    p1x = quick_predict(base)
    p2x_slider = quick_predict(apply_speed(base, 2.0))
    
    dm = copy.deepcopy(base)
    for jl in dm['judgeLineList']:
        if 'bpm' in jl: jl['bpm'] = jl['bpm'] * 2.0
    p2x_manual = quick_predict(dm)
    
    spread = p2x_slider - p1x
    print(f'  {name}: 1x={p1x:.2f}  2x_slider={p2x_slider:.2f}  2x_manual={p2x_manual:.2f}  spread={spread:+.2f} 一致={abs(p2x_slider-p2x_manual)<0.01}')

# ── 保存 ──
FLAT_F = [(feat_boost_names[j], FLAT_ORIG[j][1], float(co_current[j])) for j in range(n_boost)]
out_path = 'models/6dim_model_v7_5.pkl'
with open(out_path, 'wb') as f:
    pickle.dump({'gb': gb_final, 'scaler': sc_final, 'feature_names': FN_GB,
                 'p95_vals': P95, 'p99_vals': P99, 'FLAT_FEATURES': FLAT_F, 'dynamic_cap': DC}, f)
print(f'\n模型已保存: {out_path}')
