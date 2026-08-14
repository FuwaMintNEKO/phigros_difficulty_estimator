"""对比分析：silly/LiFE vs well-predicted 硬谱"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys,os
sys.path.insert(0,'.')
from unified_parser import load_chart
from feature_extractor import extract_features
import pickle

m=pickle.load(open(r'models\5dim_model_v4.pkl','rb'))
P95=m['p95_vals']; P99=m['p99_vals']

DL=r'C:\Users\NaNK\Downloads'
TD=os.path.join(_ROOT, 'data', 'chart', 'test_datas')

charts=[
    ('WS(17.8)',DL,'Waking Shadows(17.8).json'),
    ('CS13(17.6)',DL,'Chart_SP #1347(1).json'),
    ('silly(17.9)',DL,'silly-willy-nilly(17.9)(1).json'),
    ('LF(17.9)',TD,'LiFE Garden(1.05x).json'),
    ('FE(17.5)',DL,'Far Eastern Flavor(17.5)(1).json'),
    ('Apollo(18.0)',DL,'Apollo(18.0).json'),
    ('Submerged(18.0)',DL,'Submerged City(18.0).json'),
    ('L&J(16.7)',DL,'Love & Justice(16.7)(1).json'),
]

BURST_FEATS=[
    'total_notes','notes_per_second','tap_per_second','tap_burst_top5','tap_burst_05_top5',
    'burst_intensity_mean','peak_density_top5avg_1beat','peak_density_top5avg_0.5beat',
    'micro_peak_top5_0.0625beat','tap_micro_top5_0.0625beat','tap_micro_max_0.0625beat',
    'finger_peak_tps','finger_avg_peak_tps','hand_speed_index',
    'very_short_interval_ratio','global_jack_count','miniburst_count',
    'sustained_density_run_count','high_density_duration_ratio_16beat','high_density_duration_ratio_8beat',
    'movement_per_second','wide_jump_count','multi_finger_3plus_events',
    'density_transition_mean','density_transition_max','density_transition_std',
    'tempo_change_count','type_switch_per_sec','offbeat_ratio',
]

all_data=[]
for name,basedir,fname in charts:
    fp=os.path.join(basedir,fname)
    if not os.path.exists(fp):
        print(f'[SKIP] {name} not found')
        continue
    cd=load_chart(fp)
    fe=extract_features(cd)
    all_data.append((name,fe))

print(f'{"特征名":<38s}',end='')
for n,_ in all_data:
    print(f'{n.split("(")[0][:10]:>10s}',end='')
print()

for feat in BURST_FEATS:
    print(f'{feat:<38s}',end='')
    for _,fe in all_data:
        v=fe.get(feat,0)
        p95v=P95.get(feat,0)
        p99v=P99.get(feat,0)
        flag='!P99' if v>p99v else ('*P95' if v>p95v else '')
        if v>p95v:
            print(f'{v:>10.3f}',end='')
        else:
            print(f'{v:>10.3f}',end='')
    print()
    # Print threshold ratios for items above P95
    print(f'{"":38s}',end='')
    for _,fe in all_data:
        v=fe.get(feat,0)
        pv=P95.get(feat,0)
        thr=max(pv*0.55,0.5)
        if v>thr:
            print(f'{v/thr:>9.2f}x',end=' ')
        else:
            print(f'{"":>11s}',end='')
    print()

# Now print boost contribution
print('\n=== 各谱对总boost的贡献分解 ===')
FLAT=[
    ('peak_density_top5avg_1beat',0.5,0.20),('std_density_1beat',0.3,0.10),
    ('density_transition_mean',0.15,0.16),('density_transition_std',0.2,0.10),
    ('density_transition_max',1.0,0.08),('burst_intensity_mean',0.3,0.16),
    ('density_above_zero_ratio',0.6,0.10),
    ('notes_per_second',3.0,0.15),('tap_per_second',2.5,0.12),
    ('tap_burst_top5',0.5,0.15),('tap_micro_top5_0.0625beat',0.3,0.10),
    ('tap_micro_max_0.0625beat',0.5,0.08),('micro_peak_top5_0.0625beat',0.5,0.10),
    ('global_jack_count',20,0.07),('finger_peak_tps',2.0,0.08),
    ('total_notes',400,0.08),('tap_count',400,0.06),
    ('high_density_duration_ratio_16beat',0.05,0.10),
    ('tempo_change_count',50,0.14),('type_switch_per_sec',0.4,0.08),
    ('offbeat_ratio',0.04,0.08),('rhythm_entropy',2.5,0.06),
    ('wide_jump_density',0.5,0.08),('multi_finger_3plus_events',10,0.07),
    ('sim_pos_spread_max',3,0.07),('movement_per_second',3.0,0.06),
]

for name,fe in all_data:
    total=0
    print(f'\n--- {name} ---')
    for fn,b,c in FLAT:
        v=fe.get(fn,0); pv=P95.get(fn,0)
        t=max(pv*0.55,b*0.5)
        if v<=t: continue
        e=v/t-1; x=c*(e**0.55)
        if v>max(P99.get(fn,0),b*0.5):
            x+=c*max(0,v/max(P99.get(fn,0),b*0.5)-1)**0.55*0.5
        total+=x
        if x>0.05:
            print(f'  {fn:<35s} v={v:>8.2f} t={t:>8.3f} ratio={v/t:>5.2f}x +{x:.4f}')
    print(f'  {"总boost":<35s} {total:.4f} (capped={min(total,3.8):.4f})')
