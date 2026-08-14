import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from feature_extractor import extract_features
from predict_rpe import convert_rpe_to_standard

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', '5dim_model.pkl')

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
    d1f = [('micro_max_0.0625beat',3.0,0.80),('tap_burst_top5',10.0,0.55),('jack_count',40.0,0.40),('tap_per_second',5.5,0.40),('very_short_interval_ratio',0.30,0.35)]
    d1, t1 = _dim_boost(feats, p99, d1f, 3, 3.0)
    mf3=feats.get('multi_finger_3plus_events',0); smx=feats.get('sim_pos_spread_max',0); smn=feats.get('sim_pos_spread_mean',0.5)
    fmi=mf3*smx/max(total_n,1)*10; smi=mf3*max(1.0-smn,0)/max(total_n,1)*5
    d2=0.0; t2=0; tf=max(p99.get('multi_finger_3plus_events',30),1)*max(p99.get('sim_pos_spread_max',0.8),0.1)/max(p99.get('total_notes',500),1)*10
    if fmi>max(tf,1.0): d2=float(np.sqrt(max(float(np.log1p(fmi/max(tf,1.0)-1)),0)))/2.0; t2+=1
    if smi>0.8: d2-=0.10*min(float(np.log1p(smi)),1.0)
    d2=max(d2,-0.05)
    d3f=[('wide_jump_count',150.0,0.50),('burst_avg_movement',3.0,0.40),('hold_lock_displacement_per_sec',2.0,0.50),('hold_tap_overlap_ratio',0.5,0.25)]
    d3,t3=_dim_boost(feats,p99,d3f,2,2.0)
    d4f=[('total_notes',1200.0,0.55),('notes_per_second',9.0,0.35),('high_density_duration_ratio_16beat',0.35,0.25),('std_density_1beat',0.30,0.18)]
    d4,t4=_dim_boost(feats,p99,d4f,2,2.0); d4=min(d4,0.60)
    d5f=[('density_transition_max',4.5,0.90),('tempo_change_count',80.0,0.70),('speed_change_total_impact',80000,0.35),('offbeat_ratio',0.25,0.40),('rhythm_entropy',5.0,0.22),('bpm_change_count',3.0,0.40),('density_transition_mean',0.65,0.45)]
    d5,t5=_dim_boost(feats,p99,d5f,2,2.0)
    b=0.30*float(np.tanh((d1*0.12+d2*0.06+d3*0.15+d4*0.15+d5*0.25)/0.30))
    return min(b,0.50),{'dim1_交互纵连':round(d1,4),'dim2_多押':round(d2,4),'dim3_位移':round(d3,4),'dim4_耐力':round(d4,4),'dim5_读谱':round(d5,4),'triggers':f'{t1}/{t2}/{t3}/{t4}/{t5}'}

def predict_one(feats):
    x=np.array([[feats.get(n,0) for n in feature_names]])
    p_gb=float(gb.predict(scaler.transform(x))[0])
    pb,dims=compute_5dim_boost(feats,p95_vals,p99_vals)
    return p_gb+pb,pb,p_gb,dims

CHART_PATH = os.path.join(_ROOT, 'data', 'chart', '1321664301929799.json')

with open(CHART_PATH, 'r', encoding='utf-8') as f:
    raw = json.load(f)

meta = raw.get('META', {})
print(f'谱面: {meta.get("name", "Unknown")}')
print(f'作曲: {meta.get("composer", "Unknown")}')
print(f'制谱: {meta.get("charter", "Unknown")}')
print(f'标称等级: {meta.get("level", "Unknown")}')
print(f'BPM: {raw.get("BPMList", [{}])[0].get("bpm", "?")}')
print(f'judgeLineList: {len(raw.get("judgeLineList", []))} 条判定线')

cd = convert_rpe_to_standard(raw)
feats = extract_features(cd)

if not feats:
    print('特征提取失败!')
    exit()

pred, boost, gb_pred, dims = predict_one(feats)

# Key features
key_feats = ['total_notes','notes_per_second','tap_per_second','jack_count','duration_sec',
             'multi_finger_3plus_events','sim_pos_spread_max',
             'wide_jump_count','burst_avg_movement','hold_lock_displacement_per_sec',
             'density_transition_max','tempo_change_count','speed_change_total_impact',
             'offbeat_ratio','rhythm_entropy','bpm_change_count','bpm',
             'micro_max_0.0625beat','tap_burst_top5','bpm_range']
print('\n=== 关键特征 ===')
for k in key_feats:
    v = feats.get(k, 0)
    p99_v = p99_vals.get(k, 0)
    if v > 0:
        flag = ' ↑↑' if v > p99_v else (' ↑' if v > p99_v * 0.85 else '')
        print(f'  {k:35s}  = {v:>10.4f}  (P99={p99_v:>8.2f}){flag}')

print(f'\n=== 维度分解 ===')
print(f'  D1 交互纵连: {dims["dim1_交互纵连"]}')
print(f'  D2 多押/多指: {dims["dim2_多押"]}')
print(f'  D3 位移:     {dims["dim3_位移"]}')
print(f'  D4 耐力:     {dims["dim4_耐力"]}')
print(f'  D5 读谱:     {dims["dim5_读谱"]}')
print(f'  Triggers:    {dims["triggers"]}')

print(f'\n=== 难度预测 ===')
print(f'  GB基础:     {gb_pred:.4f}')
print(f'  +5维Boost:  {boost:.4f}')
print(f'  = 最终预测:  {pred:.4f}')
print(f'  标称等级:   {meta.get("level", "?")}')

# Compare to known charts
print(f'\n=== 与已知谱面对比 ===')
comparisons = [
    ('Rrhar\'il AT', 17.6),
    ('Igallta AT', 17.4),
    ('QZKago AT', 17.4),
    ('Distorted Fate AT', 17.4),
    ('Destruction 3,2,1 AT', 17.3),
    ('+ERABY+E AT', 17.3),
    ('AbsoluTedisoRdeR AT', 17.2),
    ('Regrets SP', 17.23),
]
for name, diff in comparisons:
    print(f'  {name:35s}  = {diff:.2f}')
print(f'  {"【哀狱炼歌】":35s}  = {pred:.2f}  ← 预测')
