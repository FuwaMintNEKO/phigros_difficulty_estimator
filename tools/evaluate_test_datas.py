import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, numpy as np
sys.path.insert(0, '.')
from feature_extractor import extract_features
from unified_parser import load_chart, extract_name
from train_5dim_v4 import compute_5dim_boost

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', '5dim_model_v4.pkl')
import pickle
with open(MODEL_PATH, 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']

TEST_DIR = os.path.join(_ROOT, 'data', 'chart', 'test_datas')
DOWNLOADS = r'C:\Users\NaNK\Downloads'

# test_datas 16张 + Downloads 4张 = 20张测试谱
# 预期值为 None 表示不需要校准该谱面的预测
test_charts = [
    # === test_datas (16张) ===
    ('CrazyTek',        TEST_DIR, '80116145.json', 17.9),
    ('Exitium',         TEST_DIR, '82991127.json', 18.5),
    ('アブノーマリティ',   TEST_DIR, '1765944423.json', None),   # 去掉预期
    ('翠杜',              TEST_DIR, '16032396.json', 17.5),
    ("DA'AT",           TEST_DIR, '2155734445357448.json', 18.2),
    ('LAMIA',           TEST_DIR, '29834645.json', 18.3),
    ('HeAskedIfIExercise', TEST_DIR, '81816997.json', 18.5),
    ('WakingShadows',   TEST_DIR, '93562988.json', 17.8),
    ('哀狱炼歌',          TEST_DIR, '1321664301929799.json', 18.6),
    ('Aether Crest',    TEST_DIR, '4641132726938698.json', 16.65),  # 16.5~16.8
    ('Chart_SP',        TEST_DIR, 'Chart_SP.json', 17.6),
    ('Chart_SP #13',    TEST_DIR, 'Chart_SP #1347(1).json', 17.6),
    ('Regrets',         TEST_DIR, 'Sigma (Haocore Mix) ~ Regrets of The Yellow Tuli.json', 17.7),
    ('105秒伝說',        TEST_DIR, 'Sigma (Haocore Mix) ~ 105秒の伝說 ~.json', 16.1),
    ('LiFE Garden(1.05x)', TEST_DIR, '6923526264684294.json', 17.9),
    ('ALL NIGHTER',     TEST_DIR, '0565845404714008.json', 16.65),  # 16.5~16.8
    # === Downloads (4张) ===
    ('People people',   DOWNLOADS, '1770391855.json', None),  # 用户说纯4k 定数不重要
    ('Galaxy Collapse', DOWNLOADS, '7009367902368871.json', None),  # 19+ 定数不重要
    ('Far Eastern Flavor', DOWNLOADS, '61901444.json', 17.5),
    ('密码的周一',       DOWNLOADS, '0582581966828779.json', 17.4),
]

print('='*100)
print('  全部测试谱面评估（训练集: 官谱957，无自定义谱）')
print('='*100)
print(f'\n{"谱面名称":<28} {"来源":>6} {"预期":>6} {"GB":>7} {"Boost":>7} {"预测":>7} {"误差":>8}')
print('-'*72)

stats_yes_exp = []  # 有预期值的
for name, base_dir, fname, expected in test_charts:
    fp = os.path.join(base_dir, fname)
    try:
        cd = load_chart(fp)
        feats = extract_features(cd)
        x = np.array([[feats.get(n, 0) for n in FN]])
        xs = scaler.transform(x)
        p_gb = float(gb.predict(xs)[0])
        p_b, dims = compute_5dim_boost(feats, P95, P99)
        p_f = p_gb + p_b
        exp_str = f'{expected:.2f}' if expected is not None else '  ?'
        err_str = f'{p_f-expected:+.3f}' if expected is not None else '  -'
        src = 'test' if base_dir == TEST_DIR else 'down'
        print(f'{name:<28} {src:>6} {exp_str:>6} {p_gb:>7.3f} {p_b:>7.3f} {p_f:>7.3f} {err_str:>8}')
        print(f'  {"":>28}  config={dims["config"]:.3f}  mf={dims["multifinger"]:.3f}  move={dims["movement"]:.3f}  endur={dims["endurance"]:.3f}  read_orig={p_b:.1f}×0.33→{dims["reading"]:.3f}')
        if expected is not None:
            stats_yes_exp.append(abs(p_f - expected))
    except Exception as e:
        import traceback
        print(f'{name:<28} 错误: {e}')

if stats_yes_exp:
    mean_abs_err = np.mean(stats_yes_exp)
    max_abs_err = np.max(stats_yes_exp)
    print(f'\n--- 有预期值的 {len(stats_yes_exp)} 张谱面 ---')
    print(f'  平均绝对误差: {mean_abs_err:.3f}')
    print(f'  最大绝对误差: {max_abs_err:.3f}')

print('='*100)
