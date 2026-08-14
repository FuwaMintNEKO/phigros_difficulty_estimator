import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import extract_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', '5dim_model.pkl')
CHART_DIR = os.path.join(_ROOT, 'data', 'chart')

with open(MODEL_PATH, 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
feature_names = m['feature_names']; p95_vals = m['p95_vals']; p99_vals = m['p99_vals']

def _dim_boost(feats, p99, feat_list, min_trig, div=2.0):
    raw = 0.0; trig_count = 0
    for fname, baseline, coeff in feat_list:
        val = feats.get(fname, 0)
        thresh = max(p99.get(fname, 0), baseline)
        if val > thresh:
            raw += coeff * float(np.log1p(val / thresh - 1)); trig_count += 1
    if trig_count == 0: return 0.0, 0
    return float(np.sqrt(raw)) * min(1.0, trig_count/max(min_trig,1)) / div, trig_count

def compute_5dim_boost(feats, p95, p99):
    total_n = max(feats.get('total_notes', 1), 1)
    d1f = [('tap_micro_max_0.0625beat',2.0,0.85),('tap_burst_top5',8.0,0.55),('jack_count',35.0,0.40),('tap_per_second',5.0,0.40),('very_short_interval_ratio',0.25,0.35)]
    d1, t1 = _dim_boost(feats, p99, d1f, 2, 2.0)
    mf3=feats.get('multi_finger_3plus_events',0); smx=feats.get('sim_pos_spread_max',0); smn=feats.get('sim_pos_spread_mean',0.5)
    fmi=mf3*smx/max(total_n,1)*10; smi=mf3*max(1.0-smn,0)/max(total_n,1)*5
    d2=0.0; t2=0; tf=max(p99.get('multi_finger_3plus_events',30),1)*max(p99.get('sim_pos_spread_max',0.8),0.1)/max(p99.get('total_notes',500),1)*10
    if fmi>max(tf,0.8): d2=float(np.sqrt(max(float(np.log1p(fmi/max(tf,0.8)-1)),0)))/1.5; t2+=1
    if smi>1.0: d2-=0.08*min(float(np.log1p(smi)),1.0)
    d2=max(d2,-0.05)
    d3f=[('wide_jump_count',120.0,0.50),('burst_avg_movement',2.5,0.40),('hold_lock_displacement_per_sec',1.5,0.50),('hold_tap_overlap_ratio',0.4,0.25)]
    d3,t3=_dim_boost(feats,p99,d3f,2,1.8)
    d4f=[('total_notes',1100.0,0.55),('tap_notes_per_second',7.5,0.40),('high_density_duration_ratio_16beat',0.30,0.25),('std_density_1beat',0.25,0.18)]
    d4,t4=_dim_boost(feats,p99,d4f,2,1.8); d4=min(d4,0.70)
    d5f=[('density_transition_max',4.0,0.90),('tempo_change_count',60.0,0.70),('speed_change_total_impact',60000,0.35),('offbeat_ratio',0.20,0.40),('rhythm_entropy',4.5,0.22),('bpm_change_count',2.0,0.40),('density_transition_mean',0.55,0.45),('type_switch_ratio',0.15,0.30)]
    d5,t5=_dim_boost(feats,p99,d5f,2,1.8)
    b=0.50*float(np.tanh((d1*0.15+d2*0.08+d3*0.15+d4*0.15+d5*0.28)/0.50))
    return min(b,0.80),{'dim1_交互纵连':round(d1,4),'dim2_多押':round(d2,4),'dim3_位移':round(d3,4),'dim4_耐力':round(d4,4),'dim5_读谱':round(d5,4),'triggers':f'{t1}/{t2}/{t3}/{t4}/{t5}'}

def predict_one(feats):
    x=np.array([[feats.get(n,0) for n in feature_names]])
    p_gb=float(gb.predict(scaler.transform(x))[0])
    pb,dims=compute_5dim_boost(feats,p95_vals,p99_vals)
    return p_gb+pb,pb,p_gb,dims

from predict_rpe import convert_rpe_to_standard

# Custom charts
custom_charts = [
    ('DA\'AT -The First Seeker of Souls-', '2155734445357448.json', 'AT Lv.17', 222),
    ('LAMIA', '29834645.json', 'AT Lv.18', 199),
    ('He asked if I exercise', '81816997.json', 'AT Lv.18', 210),
    ('Waking Shadows', '93562988.json', 'AT Lv.18', 190),
    ('哀狱炼歌', '1321664301929799.json', 'AT Lv.19', 260),
]

# SP charts for reference
sp_charts = [
    ('Chart_SP', 'Chart_SP.json'),
    ('Chart_SP #13', 'Chart_SP #1347(1).json'),
    ('Regrets', 'Sigma (Haocore Mix) ~ Regrets of The Yellow Tuli.json'),
    ('105秒伝說', 'Sigma (Haocore Mix) ~ 105秒の伝說 ~.json'),
    ('Aether Crest', '4641132726938698.json'),
]

charts_to_test = [(n, p, l, b, 'custom') for n, p, l, b in custom_charts] + \
                 [(n, p, 'SP', 0, 'sp') for n, p in sp_charts]

key_feats_list = ['total_notes','tap_count','drag_count','flick_count','duration_sec',
    'notes_per_second','tap_notes_per_second','tap_per_second',
    'jack_count','wide_jump_count','multi_finger_3plus_events',
    'tempo_change_count','speed_change_total_impact',
    'density_transition_max','offbeat_ratio','bpm_range','bpm_change_count',
    'tap_micro_max_0.0625beat','tap_burst_top5',
    'hand_speed_index','hold_lock_displacement_per_sec',
    'type_switch_ratio','rhythm_entropy',
]

print('='*110)
print(f'{"谱面名称":35s} {"标的等级":10s} {"BPM":6s} {"GB":8s} {"Boost":8s} {"预测":8s}')
print('-'*110)

all_preds = []
for name, path_suffix, level, bpm, chart_type in charts_to_test:
    fp = os.path.join(CHART_DIR, path_suffix)
    with open(fp, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    is_rpe = chart_type == 'custom' or (chart_type == 'sp' and path_suffix in ['4641132726938698.json'])
    cd = convert_rpe_to_standard(raw) if is_rpe else raw
    feats = extract_features(cd)
    if not feats: continue
    pred, boost, gb_p, dims = predict_one(feats)
    bpm_str = str(bpm) if bpm else f'{feats.get("bpm",0):.0f}'
    all_preds.append((pred, name, level, bpm_str, gb_p, boost, dims, feats))

all_preds.sort(key=lambda x: -x[0])
for pred, name, level, bpm_str, gb_p, boost, dims, feats in all_preds:
    lvl_show = f'{level:10s}' if len(level) <= 10 else f'{level:>10s}'
    print(f'{name:35s} {lvl_show} {bpm_str:6s} {gb_p:8.3f} {boost:8.4f} {pred:8.3f}')

print()
print('='*110)
print('详细5维度分解 + 关键特征')
print('='*110)

for pred, name, level, bpm_str, gb_p, boost, dims, feats in all_preds:
    print(f'\n{"="*70}')
    print(f'{name} ({level}, {bpm_str}BPM)  GB={gb_p:.3f}+Boost={boost:.4f}=预测{pred:.3f}')
    print(f'  D1交互纵连={dims["dim1_交互纵连"]}  D2多押={dims["dim2_多押"]}  D3位移={dims["dim3_位移"]}  D4耐力={dims["dim4_耐力"]}  D5读谱={dims["dim5_读谱"]}  Triggers={dims["triggers"]}')
    for k in key_feats_list:
        v = feats.get(k, 0)
        p99_v = p99_vals.get(k, 0)
        p95_v = p95_vals.get(k, 0)
        flag = ''
        if v > p99_v: flag = ' ↑↑P99'
        elif v > p95_v: flag = ' ↑P95'
        print(f'    {k:35s} = {str(v):>10s}  (P99={p99_v:>8.2f}){flag}')

print()
print('='*70)
print('官方最高难谱全量预测对比')
print('='*70)
official_high = [
    ('Rrhar\'il AT', 'Rrharil.TeamGrimoire.0', 'AT.json', 17.6),
    ('Igallta AT', 'Rrharil.TeamGrimoire.0', 'IN.json', 17.4),  # actually AT=17.6, IN=15.6
]
# Actually let's just compare with what we know from training
print('从之前训练结果看:')
print('  Rrhar\'il AT = 17.60 (官方最高)')
print('  QZKago AT = 17.40')
print('  Distorted Fate AT = 17.40')
print('  Chart_SP #13 ≈ 17.10')
print('  Regrets ≈ 17.31')
