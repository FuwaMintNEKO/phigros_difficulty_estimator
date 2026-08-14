import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from predict_rpe import convert_rpe_to_standard

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', '5dim_model.pkl')

with open(MODEL_PATH, 'rb') as f:
    m = pickle.load(f)
gb = m['gb']
scaler = m['scaler']
feature_names = m['feature_names']
p95_vals = m['p95_vals']
p99_vals = m['p99_vals']

# Copy of 5dim boost functions (self-contained, no import from train_5dim)
def _dim_boost(feats, p99, feat_list, min_trig, div=2.0):
    raw = 0.0
    trig_count = 0
    for fname, baseline, coeff in feat_list:
        val = feats.get(fname, 0)
        thresh = max(p99.get(fname, 0), baseline)
        if val > thresh:
            raw += coeff * float(np.log1p(val / thresh - 1))
            trig_count += 1
    if trig_count == 0:
        return 0.0, 0
    trig_factor = min(1.0, trig_count / max(min_trig, 1))
    return float(np.sqrt(raw)) * trig_factor / div, trig_count

def compute_5dim_boost(feats, p95, p99):
    total_n = max(feats.get('total_notes', 1), 1)
    dim1_feats = [
        ('micro_max_0.0625beat',  3.0,  0.80),
        ('tap_burst_top5',        10.0,  0.55),
        ('jack_count',            40.0,  0.40),
        ('tap_per_second',         5.5,  0.40),
        ('very_short_interval_ratio', 0.30, 0.35),
    ]
    dim1, trig1 = _dim_boost(feats, p99, dim1_feats, 3, div=3.0)
    mf3 = feats.get('multi_finger_3plus_events', 0)
    spread_max = feats.get('sim_pos_spread_max', 0)
    spread_mean = feats.get('sim_pos_spread_mean', 0.5)
    forced_mf_idx = mf3 * spread_max / max(total_n, 1) * 10
    splittable_mf_idx = mf3 * max(1.0 - spread_mean, 0) / max(total_n, 1) * 5
    dim2 = 0.0
    trig2 = 0
    thresh_fmf = max(p99.get('multi_finger_3plus_events', 30), 1) * max(p99.get('sim_pos_spread_max', 0.8), 0.1) / max(p99.get('total_notes', 500), 1) * 10
    if forced_mf_idx > max(thresh_fmf, 1.0):
        dim2 = float(np.sqrt(max(float(np.log1p(forced_mf_idx / max(thresh_fmf, 1.0) - 1)), 0))) / 2.0
        trig2 += 1
    if splittable_mf_idx > 0.8:
        dim2 -= 0.10 * min(float(np.log1p(splittable_mf_idx)), 1.0)
    dim2 = max(dim2, -0.05)
    dim3_feats = [
        ('wide_jump_count',            150.0, 0.50),
        ('burst_avg_movement',           3.0, 0.40),
        ('hold_lock_displacement_per_sec', 2.0, 0.50),
        ('hold_tap_overlap_ratio',      0.5,  0.25),
    ]
    dim3, trig3 = _dim_boost(feats, p99, dim3_feats, 2, div=2.0)
    dim4_feats = [
        ('total_notes',               1200.0, 0.55),
        ('notes_per_second',           9.0,   0.35),
        ('high_density_duration_ratio_16beat', 0.35, 0.25),
        ('std_density_1beat',          0.30,  0.18),
    ]
    dim4, trig4 = _dim_boost(feats, p99, dim4_feats, 2, div=2.0)
    dim4 = min(dim4, 0.60)
    dim5_feats = [
        ('density_transition_max',      4.5,  0.90),
        ('tempo_change_count',         80.0,  0.70),
        ('speed_change_total_impact', 80000,  0.35),
        ('offbeat_ratio',              0.25,  0.40),
        ('rhythm_entropy',             5.0,   0.22),
        ('bpm_change_count',            3.0,  0.40),
        ('density_transition_mean',     0.65, 0.45),
    ]
    dim5, trig5 = _dim_boost(feats, p99, dim5_feats, 2, div=2.0)
    raw_total = dim1 * 0.12 + dim2 * 0.06 + dim3 * 0.15 + dim4 * 0.15 + dim5 * 0.25
    cap = 0.30
    total_boost = cap * float(np.tanh(raw_total / cap))
    total_boost = min(total_boost, 0.50)
    return total_boost, {'dim1_交互纵连': round(dim1, 3), 'dim2_多押': round(dim2, 3),
                          'dim3_位移': round(dim3, 3), 'dim4_耐力': round(dim4, 3),
                          'dim5_读谱': round(dim5, 3),
                          'triggers': f'{trig1}/{trig2}/{trig3}/{trig4}/{trig5}'}

def predict_one(feats):
    x = np.array([[feats.get(n, 0) for n in feature_names]])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    p_boost, dims = compute_5dim_boost(feats, p95_vals, p99_vals)
    return p_gb + p_boost, p_boost, p_gb, dims

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)

all_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    for lv in ['EZ','HD','IN','AT']:
        if lv in info['levels'] and lv in diffs:
            all_items.append({'folder':fn,'filepath':info['levels'][lv],'difficulty':diffs[lv],'level':lv})

special_charts = [
    ('Chart_SP', os.path.join(_ROOT, 'data', 'chart', 'Chart_SP.json'), False),
    ('Chart_SP #13', os.path.join(_ROOT, 'data', 'chart', 'Chart_SP #1347(1).json'), False),
    ('Regrets', os.path.join(_ROOT, 'data', 'chart', 'Sigma (Haocore Mix) ~ Regrets of The Yellow Tuli.json'), False),
    ('105秒伝說', os.path.join(_ROOT, 'data', 'chart', 'Sigma (Haocore Mix) ~ 105秒の伝說 ~.json'), False),
    ('Aether Crest', os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json'), True),
]

results = []

for item in all_items:
    if item['difficulty'] < 16.0:
        continue
    try:
        cd = load_chart_json(item['filepath'])
        feats = extract_features(cd)
        if not feats: continue
        pred, boost, gb_pred, dims = predict_one(feats)
        name = item['folder'].replace('.0', '')
        results.append({
            'name': name, 'level': item['level'], 'true': item['difficulty'],
            'pred': pred, 'gb': gb_pred, 'boost': boost,
            'd1': dims['dim1_交互纵连'], 'd2': dims['dim2_多押'],
            'd3': dims['dim3_位移'], 'd4': dims['dim4_耐力'], 'd5': dims['dim5_读谱'],
            'triggers': dims.get('triggers', ''),
            'source': 'dataset',
        })
    except: pass

for name, path, is_rpe in special_charts:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        cd = convert_rpe_to_standard(raw) if is_rpe else raw
        feats = extract_features(cd)
        if not feats: continue
        pred, boost, gb_pred, dims = predict_one(feats)
        meta = raw.get('META', {})
        level_info = meta.get('level', 'SP')
        results.append({
            'name': name, 'level': level_info, 'true': None,
            'pred': pred, 'gb': gb_pred, 'boost': boost,
            'd1': dims['dim1_交互纵连'], 'd2': dims['dim2_多押'],
            'd3': dims['dim3_位移'], 'd4': dims['dim4_耐力'], 'd5': dims['dim5_读谱'],
            'triggers': dims.get('triggers', ''),
            'source': 'SP',
        })
    except: pass

results.sort(key=lambda r: -r['pred'])

print('=' * 130)
print('  Phigros 16.0+ 谱面 + 特殊SP谱面 — 5维度难度分解')
print(f'  共 {len(results)} 个谱面')
print('=' * 130)
print(f'{"#":>3s}  {"谱面名称":<36s}  {"Lv":>3s}  {"难度":>5s}  {"GB":>5s}  {"+B":>4s}'
      f'  {"D1交互":>6s}  {"D2多押":>6s}  {"D3位移":>6s}  {"D4耐力":>6s}  {"D5读谱":>6s}  {"trig":>10s}  {"误差":>7s}')
print('-' * 130)

for rank, r in enumerate(results, 1):
    name = r['name'][:36]
    true_str = f'{r["true"]:.1f}' if r['true'] is not None else '  SP '
    err_str = f'{r["pred"]-r["true"]:+.3f}' if r['true'] is not None else '   SP'
    print(f'{rank:>3d}  {name:<36s}  {r["level"]:>3s}  {true_str:>5s}'
          f'  {r["gb"]:>5.2f}  {r["boost"]:>4.3f}'
          f'  {r["d1"]:>6.3f}  {r["d2"]:>6.3f}  {r["d3"]:>6.3f}  {r["d4"]:>6.3f}  {r["d5"]:>6.3f}'
          f'  {r["triggers"]:>10s}  {err_str:>7s}')

print('=' * 130)
at_items = [r for r in results if r['level'] == 'AT' and r['true'] is not None]
if at_items:
    print(f'  AT (n={len(at_items)}): MAE={np.mean([abs(r["pred"]-r["true"]) for r in at_items]):.3f}, '
          f'偏差={np.mean([r["pred"]-r["true"] for r in at_items]):+.3f}')
print(f'  全部高难(n={len(results)}): MAE={np.mean([abs(r["pred"]-r["true"]) for r in results if r["true"] is not None]):.3f}')
print('=' * 130)
