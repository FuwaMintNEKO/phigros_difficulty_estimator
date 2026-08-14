import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, copy, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', '5dim_model_v3.pkl')
CHART_DIR = os.path.join(_ROOT, 'data', 'chart')

with open(MODEL_PATH, 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
feature_names = m['feature_names']; p95_vals = m['p95_vals']; p99_vals = m['p99_vals']

def _compute_dim_boost(feats, p95, p99, feat_list):
    raw = 0.0
    for fname, baseline, coeff in feat_list:
        val = feats.get(fname, 0)
        thresh = max(p95.get(fname, 0), baseline)
        if val <= thresh: continue
        excess = val / thresh - 1.0
        contrib = coeff * (excess ** 0.6)
        if val > max(p99.get(fname, 0), baseline):
            p99_excess = val / max(p99.get(fname, 0), baseline) - 1.0
            p99_bonus = coeff * max(0, p99_excess) ** 0.6 * 0.5
            contrib += p99_bonus
        raw += contrib
    return raw

def compute_5dim_boost(feats, p95, p99):
    total_n = max(feats.get('total_notes', 1), 1)
    d1f = [('tap_micro_max_0.0625beat',2.0,0.55),('tap_micro_top5_0.0625beat',1.2,0.40),('tap_burst_top5',6.0,0.35),('jack_count',20.0,0.30),('tap_per_second',4.2,0.30),('very_short_interval_ratio',0.18,0.25),('tap_burst_05_top5',4.0,0.35)]
    d1 = _compute_dim_boost(feats,p95,p99,d1f)
    mf3=feats.get('multi_finger_3plus_events',0); smx=feats.get('sim_pos_spread_max',0)
    fmi=mf3*smx/max(total_n,1)*10; d2=0.0
    th=max(p99.get('multi_finger_3plus_events',30),1)*max(p99.get('sim_pos_spread_max',0.8),0.1)/max(p99.get('total_notes',500),1)*10
    if fmi>max(th*0.5,0.3): d2=0.50*((fmi/max(th*0.5,0.3)-1)**0.6)
    d3f=[('wide_jump_count',60.0,0.40),('burst_avg_movement',1.5,0.30),('hold_lock_displacement_per_sec',0.8,0.40),('movement_per_second',7.0,0.12),('hold_lock_tap_events_per_hold',1.0,0.25)]
    d3=_compute_dim_boost(feats,p95,p99,d3f)
    d4f=[('total_notes',800.0,0.45),('tap_notes_per_second',5.0,0.35),('notes_per_second',7.5,0.15),('high_density_duration_ratio_16beat',0.15,0.20),('sustained_density_run_count',1.0,0.18)]
    d4=_compute_dim_boost(feats,p95,p99,d4f)
    d5f=[('density_transition_max',2.5,0.75),('tempo_change_count',30.0,0.55),('speed_change_total_impact',20000,0.28),('offbeat_ratio',0.08,0.30),('rhythm_entropy',3.0,0.15),('bpm_change_count',0.5,0.30),('density_transition_mean',0.30,0.38),('type_switch_ratio',0.06,0.22),('type_switch_per_sec',0.8,0.18)]
    d5=_compute_dim_boost(feats,p95,p99,d5f)
    tb=d1*0.22+d2*0.10+d3*0.18+d4*0.18+d5*0.30
    return min(tb,3.0),{'dim1_交互纵连':round(d1,4),'dim2_多押':round(d2,4),'dim3_位移':round(d3,4),'dim4_耐力':round(d4,4),'dim5_读谱':round(d5,4)}

def predict_one(feats):
    x=np.array([[feats.get(n,0) for n in feature_names]])
    p_gb=float(gb.predict(scaler.transform(x))[0])
    pb,dims=compute_5dim_boost(feats,p95_vals,p99_vals)
    return p_gb+pb,pb,p_gb,dims

from predict_rpe import convert_rpe_to_standard

# 新自制谱 (不在训练集中)
new_charts = [
    ('LiFE Garden(1.05x)', '6923526264684294.json', 'AT Lv.18', 156.45),
]

# 已训练的谱面参考
trained_custom = [
    ('哀狱炼歌(训练集)', '1321664301929799.json'),
    ('DA\'AT(训练集)', '2155734445357448.json'),
    ('LAMIA(训练集)', '29834645.json'),
    ('HeAsked(训练集)', '81816997.json'),
    ('WakingShadows(训练集)', '93562988.json'),
]

# 官方最高难参考
official_high = [
    ('Rrhar\'il AT', 'Rrharil.TeamGrimoire.0', 'AT.json'),
    ('QZKago AT', 'QZKagoRequiem.tpazolite.0', 'AT.json'),
]

# 测试新谱
print('='*70)
print('  【未见过的】新谱面测试')
print('='*70)
for name, path_suffix, level, bpm in new_charts:
    fp = os.path.join(CHART_DIR, path_suffix)
    with open(fp, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    cd = convert_rpe_to_standard(raw)
    feats = extract_features(cd)
    if not feats: continue
    pred, boost, gb_p, dims = predict_one(feats)

    print(f'\n{name} ({level}, {bpm}BPM) — 未参与训练！')
    print(f'  GB基础: {gb_p:.3f}')
    print(f'  +5维Boost: {boost:.4f}')
    print(f'  = 预测: {pred:.3f}')
    print(f'  标称AT Lv.18')
    for k, v in dims.items():
        print(f'  {k}: {v}')

    # 关键特征
    for k in ['total_notes','tap_count','drag_count','duration_sec','notes_per_second',
              'tap_notes_per_second','jack_count','wide_jump_count','multi_finger_3plus_events',
              'tempo_change_count','density_transition_max','offbeat_ratio','bpm_change_count',
              'tap_micro_max_0.0625beat','tap_burst_top5','type_switch_ratio']:
        v = feats.get(k, 0)
        p99_v = p99_vals.get(k, 0)
        flag = ' ↑↑' if v > p99_v else (' ↑' if v > p99_v * 0.85 else '')
        if v > 0:
            print(f'    {k:35s} = {str(v):>10s}  (P99={p99_v:>8.2f}){flag}')

# 训练集参考
print('\n' + '='*70)
print('  训练集谱面参考（模型已记住）')
print('='*70)
for name, path_suffix in trained_custom:
    fp = os.path.join(CHART_DIR, path_suffix)
    with open(fp, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    cd = convert_rpe_to_standard(raw)
    feats = extract_features(cd)
    if not feats: continue
    pred, boost, gb_p, dims = predict_one(feats)
    print(f'  {name:30s}  GB={gb_p:.3f} +Boost={boost:.3f} = {pred:.3f}')

# 官方高难
print('\n' + '='*70)
print('  官方高难谱面参考')
print('='*70)
for name, folder, json_file in official_high:
    fp = os.path.join(CHART_DIR, folder, json_file)
    with open(fp, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    feats = extract_features(raw)
    if not feats: continue
    pred, boost, gb_p, dims = predict_one(feats)
    print(f'  {name:30s}  GB={gb_p:.3f} +Boost={boost:.3f} = {pred:.3f}')
