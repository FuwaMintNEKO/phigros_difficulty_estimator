import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys; sys.path.insert(0, '.')
import os, pickle, numpy as np
from feature_extractor import extract_features
from unified_parser import load_chart

MP = os.path.join(os.path.dirname(__file__), 'models', '5dim_model_v4.pkl')
with open(MP, 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']; FN = m['feature_names']
P95 = m['p95_vals']; P99 = m['p99_vals']

def _cdb(f,p95,p99,fl):
    r=0.0
    for fn,b,c in fl:
        v=f.get(fn,0); t=max(p95.get(fn,0),b)
        if v<=t: continue
        e=v/t-1; x=c*(e**0.6)
        if v>max(p99.get(fn,0),b): x+=c*max(0,v/max(p99.get(fn,0),b)-1)**0.6*0.5
        r+=x
    return r

def cb(fe):
    tn=max(fe.get('total_notes',1),1)
    d1=_cdb(fe,P95,P99,[('tap_micro_max_0.0625beat',1.0,0.90),('tap_micro_top5_0.0625beat',0.8,0.70),('tap_burst_top5',4.0,0.65),('short_jack_count',15.0,0.22),('long_jack_count',10.0,0.26),('jack_max_run',5.0,0.22),('tap_per_second',3.5,0.70),('very_short_interval_ratio',0.12,0.55),('tap_burst_05_top5',3.0,0.55),('finger_peak_tps',3.0,0.70),('finger_avg_peak_tps',2.0,0.45)])
    mf=fe.get('multi_finger_3plus_events',0);sm=fe.get('sim_pos_spread_max',0);fmi=mf*sm/max(tn,1)*10;d2=0.0
    th=max(P99.get('multi_finger_3plus_events',30),1)*max(P99.get('sim_pos_spread_max',0.8),0.1)/max(P99.get('total_notes',500),1)*10
    if fmi>max(th*0.3,0.2): d2=1.20*((fmi/max(th*0.3,0.2)-1)**0.55)
    d2+=_cdb(fe,P95,P99,[('cross_line_3plus_count',3.0,0.80),('multi_line_sim_ratio',0.08,0.50),('stair_total_steps',20.0,0.50)])*0.35
    d2+=_cdb(fe,P95,P99,[('avg_chord_size',2.0,0.30)])*0.10
    d3=_cdb(fe,P95,P99,[('wide_jump_count',30.0,0.70),('burst_avg_movement',1.0,0.50),('hold_lock_displacement_per_sec',0.5,0.60),('movement_per_second',5.0,0.35),('stair_event_count',3.0,0.35)])
    d4=_cdb(fe,P95,P99,[('total_notes',500.0,0.70),('tap_notes_per_second',3.5,0.55),('notes_per_second',5.0,0.30),('high_density_duration_ratio_16beat',0.10,0.35),('sustained_density_run_count',0.5,0.30)])
    d5r=_cdb(fe,P95,P99,[('density_transition_max',2.5,0.85),('tempo_change_count',30.0,0.60),('speed_change_total_impact',20000,0.30),('offbeat_ratio',0.08,0.30),('rhythm_entropy',3.0,0.18),('bpm_change_count',0.5,0.30),('density_transition_mean',0.30,0.50),('type_switch_ratio',0.06,0.30),('type_switch_per_sec',0.8,0.25)])
    d5=2.5*d5r/max(d5r+0.6,0.01)
    tb=d1*0.25+d2*0.25+d3*0.18+d4*0.08+d5*0.14
    return min(tb,8.0),{'d1':d1,'d2':d2,'d3':d3,'d4':d4,'d5':d5}

DOWNLOADS = r'C:\Users\NaNK\Downloads'
new_charts = [
    ('Apollo', 'Apollo(18.0).json', 18.0),
    ('Love & Justice', 'Love & Justice(16.7)(1).json', 16.7),
    ('Xaleid◆scopiX', 'Xaleid◆scopiX(18.2)(1).json', 18.2),
]

print(f'{"谱面":<20} {"预期":>6} {"GB":>8} {"Boost":>7} {"预测":>7} {"误差":>8}')
print('-'*60)
for name, fname, exp in new_charts:
    fp = os.path.join(DOWNLOADS, fname)
    cd = load_chart(fp); fe = extract_features(cd)
    x = np.array([[fe.get(n,0) for n in FN]]); xs = scaler.transform(x)
    gv = float(gb.predict(xs)[0]); bv, ds = cb(fe); p = gv + bv
    err = p - exp
    print(f'{name:<20} {exp:>6.1f} {gv:>8.3f} {bv:>7.3f} {p:>7.3f} {err:>+8.3f}')
    print(f'  d1={ds["d1"]:.3f} d2={ds["d2"]:.3f} d3={ds["d3"]:.3f} d4={ds["d4"]:.3f} d5={ds["d5"]:.3f}')
