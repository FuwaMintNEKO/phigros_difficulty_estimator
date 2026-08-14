import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, copy, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_chart_json
from feature_extractor import extract_features

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

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')

# ---- Rrhar'il ----
chart_path = os.path.join(CHART_DIR, 'Rrharil.TeamGrimoire.0', 'AT.json')

with open(chart_path, 'r', encoding='utf-8') as f:
    original = json.load(f)

judge_lines = original.get('judgeLineList', [])
print(f'=== Rrharil AT: {len(judge_lines)} judgeLines ===')
for i, line in enumerate(judge_lines[:3]):
    lb = line.get('bpm', 'NOT_FOUND')
    be = line.get('bpmEvents', [])
    print(f'  Line {i}: bpm={lb}, bpmEvents={len(be)}')

modified = copy.deepcopy(original)
mod_count = 0
for line in modified.get('judgeLineList', []):
    if 'bpm' in line:
        line['bpm'] = round(line['bpm'] * 1.1, 4)
        mod_count += 1
    for ev in line.get('bpmEvents', []):
        if 'bpm' in ev:
            ev['bpm'] = round(ev['bpm'] * 1.1, 4)
if 'META' in modified and 'BPM' in modified['META']:
    modified['META']['BPM'] = round(modified['META']['BPM'] * 1.1, 4)

feats_orig = extract_features(original)
feats_mod = extract_features(modified)
pred_orig, boost_orig, gb_orig, dims_orig = predict_one(feats_orig)
pred_mod, boost_mod, gb_mod, dims_mod = predict_one(feats_mod)

print(f'\n修改了 {mod_count} 个judgeLine的bpm')

print('\n=== 关键特征变化 ===')
for k in ['bpm','notes_per_second','tap_per_second','tap_notes_per_second','duration_sec','total_notes','jack_count','multi_finger_3plus_events','tempo_change_count','speed_change_total_impact','bpm_change_count']:
    vo=feats_orig.get(k,0); vm=feats_mod.get(k,0)
    if abs(vm-vo)>1e-8:
        print(f'  {k:35s}  {vo:>10.4f}  ->  {vm:>10.4f}  ({vm-vo:+10.4f})')

print('\n=== 所有变化特征 ===')
for k in sorted(feature_names):
    vo=feats_orig.get(k,0); vm=feats_mod.get(k,0)
    if abs(vm-vo)>1e-8:
        print(f'  {k:35s}  {vo:>10.4f}  ->  {vm:>10.4f}  ({vm-vo:+10.4f})')

print('\n' + '='*60)
print('  Rrharil AT BPM×1.1')
print('='*60)
print(f'  GB:     {gb_orig:.4f} -> {gb_mod:.4f} ({gb_mod-gb_orig:+.4f})')
print(f'  Boost:  {boost_orig:.4f} -> {boost_mod:.4f} ({boost_mod-boost_orig:+.4f})')
print(f'  最终:   {pred_orig:.4f} -> {pred_mod:.4f} ({pred_mod-pred_orig:+.4f})')
for k in ['dim1_交互纵连','dim2_多押','dim3_位移','dim4_耐力','dim5_读谱']:
    print(f'  {k}: {dims_orig[k]} -> {dims_mod[k]}')
print(f'  Triggers: {dims_orig["triggers"]} -> {dims_mod["triggers"]}')
print('='*60)
