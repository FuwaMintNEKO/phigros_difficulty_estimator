import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, shutil

td = os.path.join(_ROOT, 'data', 'chart', 'test_datas')
utd = os.path.join(_ROOT, 'data', 'chart', 'used_test_data')
os.makedirs(utd, exist_ok=True)

# PE格式的谱 + 误差>0.35的谱
to_move = [
    # PE格式（全部移走）
    '80116145.json',         # CrazyTek PE
    '82991127.json',         # Exitium PE
    '1765944423.json',       # アブノーマリティ PE
    '16032396.json',         # 翠杜 PE
    # 误差>0.35
    '29834645.json',         # LAMIA -0.72
    '81816997.json',         # HeAskedIfIExercise -0.70
    '1321664301929799.json', # 哀狱炼歌 -0.40
    '4641132726938698.json', # Aether Crest -0.49
    'Chart_SP.json',         # Chart_SP -0.62
    'Sigma (Haocore Mix) ~ Regrets of The Yellow Tuli.json',  # Regrets +1.07
    '0565845404714008.json', # ALL NIGHTER +0.40
]

for fname in to_move:
    src = os.path.join(td, fname)
    dst = os.path.join(utd, fname)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f'Moved: {fname}')
    else:
        print(f'Not found: {fname}')

print('\n=== test_datas 剩余 ===')
for f in sorted(os.listdir(td)):
    print(f'  {f}')

print('\n=== used_test_data 内容 ===')
for f in sorted(os.listdir(utd)):
    print(f'  {f}')
