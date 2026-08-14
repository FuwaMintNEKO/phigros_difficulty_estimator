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
]

for name, fname, diff in new_files:
    fp = os.path.join(DOWNLOADS, fname)
    print(f'\n{"="*60}')
    print(f'{name} (预期={diff})')
    print(f'文件: {fname}')
    
    # 检查格式
    with open(fp, encoding='utf-8') as f:
        content = f.read()
    
    # 判断格式
    stripped = content.strip()
    if stripped.startswith('{'):
        parsed = json.loads(stripped)
        if 'META' in parsed:
            fmt = 'RPE'
        elif 'judgeLineList' in parsed:
            fmt = '标准JSON'
        else:
            fmt = '未知JSON'
    elif any(stripped.startswith(p) for p in ['n1', 'n2', 'n3', 'n4', 'bp', 'fi', 'id']):
        fmt = 'PE'
    else:
        fmt = f'未知(前50字符:{stripped[:50]})'
    print(f'格式: {fmt}')
    
    try:
        cd = load_chart(fp)
        fe = extract_features(cd)
        print(f'  total_notes={fe["total_notes"]}  tap={fe["tap_count"]}')
        print(f'  tap_per_second={fe["tap_per_second"]:.2f}  notes_per_second={fe["notes_per_second"]:.2f}')
        print(f'  multi_finger_3plus={fe.get("multi_finger_3plus_events",0)}')
        print(f'  wide_jump_count={fe.get("wide_jump_count",0)}')
    except Exception as e:
        print(f'  加载失败: {e}')
