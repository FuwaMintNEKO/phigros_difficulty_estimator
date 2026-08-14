import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'gb_final_model.pkl')

with open(MODEL_PATH, 'rb') as f:
    m = pickle.load(f)
gb = m['gb']
scaler = m['scaler']
feature_names = m['feature_names']
p95_vals = m['p95_vals']

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)

all_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    diffs = song_difficulties[sid]
    for lv in ['AT']:
        if lv in info['levels'] and lv in diffs:
            all_items.append({'name': fn, 'filepath': info['levels'][lv],
                              'difficulty': diffs[lv], 'level': lv})

# 加入特殊测试谱面
from predict_rpe import convert_rpe_to_standard

special_charts = [
    ('Chart_SP', os.path.join(_ROOT, 'data', 'chart', 'Chart_SP.json'), False, None),
    ('Chart_SP #13', os.path.join(_ROOT, 'data', 'chart', 'Chart_SP #1347(1).json'), False, None),
    ('Regrets', os.path.join(_ROOT, 'data', 'chart', 'Sigma (Haocore Mix) ~ Regrets of The Yellow Tuli.json'), False, None),
    ('105秒伝說', os.path.join(_ROOT, 'data', 'chart', 'Sigma (Haocore Mix) ~ 105秒の伝說 ~.json'), False, None),
    ('Aether Crest', os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json'), True, None),
]

def compute_boost_formula(feats):
    total_n = max(feats.get('total_notes', 1), 1)
    
    mf_ratio = feats.get('multi_finger_3plus_events', 0) / total_n * 20
    mf_eps = feats.get('mf_events_per_second', 0) * 3
    speed_n = min(feats.get('speed_change_total_impact', 0) / 50000.0, 3.0)
    keyboard_idx = mf_ratio + mf_eps + speed_n
    
    hold_lock_r = feats.get('hold_lock_tap_events', 0) / total_n * 10
    hold_ov = feats.get('hold_tap_overlap_ratio', 0) * 3
    dt_mean = feats.get('density_transition_mean', 0) / 0.65 * 1.5
    burst_mv = feats.get('burst_avg_movement', 0) / 5.0
    awkward_idx = hold_lock_r + hold_ov + dt_mean + burst_mv
    
    smooth = (awkward_idx + 1.0) / (awkward_idx + keyboard_idx + 1.0)
    smooth = float(np.clip(smooth, 0.40, 0.85))
    
    kb_baselines = [
        ('multi_finger_3plus_events', 30.0, 0.09),
        ('wide_jump_count',          250.0, 0.09),
        ('micro_max_0.0625beat',     4.0, 0.12),
        ('notes_per_second',         10.0, 0.05),
        ('tap_per_second',           5.5, 0.04),
    ]
    kb_boost = 0.0
    for fname, baseline, coeff in kb_baselines:
        val = feats.get(fname, 0)
        p95_val = max(p95_vals.get(fname, 0), baseline)
        if val > p95_val:
            kb_boost += coeff * float(np.log1p(val / p95_val - 1))
    kb_boost *= smooth
    
    ak_baselines = [
        ('density_transition_max',   4.0, 0.33),
        ('max_concurrent_holds',     3.0, 0.06),
        ('hold_lock_displacement_per_sec', 3.0, 0.13),
        ('hold_tap_overlap_ratio',   0.6, 0.12),
    ]
    ak_boost = 0.0
    for fname, baseline, coeff in ak_baselines:
        val = feats.get(fname, 0)
        p95_val = max(p95_vals.get(fname, 0), baseline)
        if val > p95_val:
            ak_boost += coeff * float(np.log1p(val / p95_val - 1))
    
    dom_r = feats.get('dominant_rhythm_ratio', 0.3)
    rh_ent = feats.get('rhythm_entropy', 2.5)
    smooth_penalty = 0.0
    if dom_r > 0.38:
        smooth_penalty -= (dom_r - 0.38) * 0.5
    if rh_ent < 2.2:
        smooth_penalty -= (2.2 - rh_ent) * 0.3
    smooth_penalty = max(smooth_penalty, -0.30)
    
    return min(kb_boost + ak_boost + smooth_penalty, 1.5), smooth

results = []

print(f'处理 {len(all_items)} 张AT谱面...')
for i, item in enumerate(all_items):
    try:
        cd = load_chart_json(item['filepath'])
        feats = extract_features(cd)
        if not feats: continue
        x = np.array([[feats.get(n, 0) for n in feature_names]])
        xs = scaler.transform(x)
        p_gb = float(gb.predict(xs)[0])
        boost, smooth = compute_boost_formula(feats)
        p_final = p_gb + boost
        err = p_final - item['difficulty']
        results.append({
            'name': item['name'], 'level': item['level'], 'true': item['difficulty'],
            'gb': p_gb, 'boost': boost, 'pred': p_final, 'err': err, 'smooth': smooth,
            'is_special': False,
        })
    except Exception as e:
        print(f'  ERR {item["name"]}: {e}')
    if (i+1)%20==0: print(f'  {i+1}/{len(all_items)}')

print(f'\n处理特殊谱面...')
for name, path, is_rpe, _ in special_charts:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        cd = convert_rpe_to_standard(raw) if is_rpe else raw
        feats = extract_features(cd)
        if not feats: continue
        x = np.array([[feats.get(n, 0) for n in feature_names]])
        xs = scaler.transform(x)
        p_gb = float(gb.predict(xs)[0])
        boost, smooth = compute_boost_formula(feats)
        p_final = p_gb + boost
        results.append({
            'name': name, 'level': '-', 'true': None,
            'gb': p_gb, 'boost': boost, 'pred': p_final, 'err': None,
            'smooth': smooth, 'is_special': True,
        })
    except Exception as e:
        print(f'  ERR {name}: {e}')

# 按预测难度排序
results.sort(key=lambda x: -x['pred'])

total_n = len(results)
at_only = [r for r in results if not r['is_special'] and r['level'] == 'AT']
specials = [r for r in results if r['is_special']]

print(f'\n{"="*90}')
print(f'  AT谱面 + 特殊测试谱面（按预测难度排序）')
print(f'{"="*90}')
print(f'{"":3s} {"曲名":36s} {"标注":6s} {"GB":6s} {"Boost":6s} {"预测":7s} {"真值":6s} {"误差":6s}')
print(f'{"-"*90}')

for i, r in enumerate(results):
    name = r['name'] if len(r['name']) <= 36 else r['name'][:33] + '...'
    true_str = f'{r["true"]:6.1f}' if r['true'] is not None else '   ?  '
    lv = r['level']
    if r['is_special']:
        name = f'** {name}'
        lv = 'TEST'
    
    if r['err'] is None:
        err_str = '   ?  '
        emoji = '❓'
    else:
        err_str = f'{r["err"]:+6.3f}'
        emoji = '✅' if abs(r['err']) <= 0.3 else ('⚠️' if abs(r['err']) <= 0.5 else '❌')
    
    print(f'  {i+1:3d} {name:36s} {lv:6s} {r["gb"]:6.2f} {r["boost"]:+6.3f} {r["pred"]:7.2f} {true_str} {err_str} {emoji}')

# 汇总
abs_errs = [abs(r['err']) for r in at_only]
errs = [r['err'] for r in at_only]
n = len(at_only)
print(f'\n  AT汇总: n={n}, MAE={np.mean(abs_errs):.3f}, 偏差={np.mean(errs):+.3f}')
print(f'    正误差(预测>真值): {sum(1 for e in errs if e > 0)}')
print(f'    负误差(预测<真值): {sum(1 for e in errs if e < 0)}')
print(f'    ±0.1: {sum(1 for e in abs_errs if e<=0.1)/n*100:.0f}%')
print(f'    ±0.3: {sum(1 for e in abs_errs if e<=0.3)/n*100:.0f}%')
print(f'    ±0.5: {sum(1 for e in abs_errs if e<=0.5)/n*100:.0f}%')

print(f'\n  特殊谱面:')
for r in specials:
    print(f'    {r["name"]:30s} 预测={r["pred"]:.2f} (GB={r["gb"]:.2f}, boost={r["boost"]:.3f}, smooth={r["smooth"]:.3f})')
