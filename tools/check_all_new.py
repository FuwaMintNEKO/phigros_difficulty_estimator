import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys; sys.path.insert(0, '.')
import os, json
from unified_parser import load_chart
from feature_extractor import extract_features

DOWNLOADS = r'C:\Users\NaNK\Downloads'

new_files = [
    ('Apollo', 'Apollo(18.0).json', 18.0),
    ('Love & Justice', 'Love & Justice(16.7)(1).json', 16.7),
    ('Xaleid◆scopiX', 'Xaleid◆scopiX(18.2)(1).json', 18.2),
    ('silly-willy-nilly', 'silly-willy-nilly(17.9)(1).json', 17.9),
    ('おぎゃりないざー', 'おぎゃりないざー(16.5~16.6).json', 16.55),
    ('恋ひ恋ふ縁', '恋ひ恋ふ縁(16.8)(1).json', 16.8),
    ('朧月', '朧月(18.4)(1).json', 18.4),
    ('天方地園', '天方地園(16.9)(1).json', 16.9),
    ('ニャンだふる♡サマー!!', 'ニャンだふる♡サマー!!(15.8).json', 15.8),
]

for name, fname, diff in new_files:
    fp = os.path.join(DOWNLOADS, fname)
    print(f'\n=== {name} (预期={diff}) ===')
    
    try:
        cd = load_chart(fp)
        fe = extract_features(cd)
        print(f'  total_notes={fe["total_notes"]}  tap={fe["tap_count"]}  hold={fe.get("hold_count",0)}')
        print(f'  tap_per_second={fe["tap_per_second"]:.2f}  notes_per_second={fe["notes_per_second"]:.2f}')
        print(f'  multi_finger_3plus={fe.get("multi_finger_3plus_events",0)}')
        print(f'  wide_jump={fe.get("wide_jump_count",0)}')
        print(f'  finger_peak_tps={fe.get("finger_peak_tps",0):.1f}')
        print(f'  finger_avg_peak_tps={fe.get("finger_avg_peak_tps",0):.1f}')
    except Exception as e:
        print(f'  ERROR: {e}')
