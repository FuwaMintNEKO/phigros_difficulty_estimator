import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os
sys.path.insert(0, '.')
from unified_parser import load_chart
from feature_extractor import extract_features
import pickle

m = pickle.load(open(r'models\5dim_model_v4.pkl', 'rb'))
P95 = m['p95_vals']
P99 = m['p99_vals']

DL = r'C:\Users\NaNK\Downloads'
targets = [
    ('密码的周一', DL, '0582581966828779.json'),
    ('恋ひ恋ふ縁', DL, '恋ひ恋ふ縁(16.8)(1).json'),
    ('天方地園', DL, '天方地園(16.9)(1).json'),
    ('666', DL, '666(16.5).json'),
    ("Angel's Salad", DL, "Angel's Salad(16.9).json"),
    ('Breakcore革命前夜', DL, 'Breakcore革命前夜(16.3~16.5).json'),
    ('Lemegeton', DL, 'Lemegeton -little key of solomon-(16.6).json'),
]

KEYS = ['total_notes', 'notes_per_second', 'tap_per_second',
        'multi_finger_3plus_events', 'tempo_change_count', 'offbeat_ratio',
        'density_transition_max', 'tap_micro_max_0.0625beat', 'tap_burst_top5',
        'wide_jump_count', 'very_short_interval_ratio', 'hold_count',
        'speed_change_total_impact', 'jack_count', 'same_line_jack_count']

for name, basedir, fname in targets:
    fp = os.path.join(basedir, fname)
    cd = load_chart(fp)
    fe = extract_features(cd)
    print(f'=== {name} ===')
    for k in KEYS:
        v = fe.get(k, 0)
        p95v = P95.get(k, 0)
        p99v = P99.get(k, 0)
        flag = '>P99' if v > p99v else ('>P95' if v > p95v else '')
        print(f'  {k:35s} = {str(v):>10s}  P95={str(p95v):>8s} P99={str(p99v):>8s}  {flag}')
    print()
