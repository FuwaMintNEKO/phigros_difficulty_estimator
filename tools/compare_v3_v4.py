import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import os, sys, json, pickle, numpy as np
sys.path.insert(0, r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator')
from unified_parser import load_chart
from feature_extractor import extract_features

TD = os.path.join(_ROOT, 'data', 'chart', 'test_datas')
DL = r'C:\Users\NaNK\Downloads'

# Load v3 model
with open(r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator\models\5dim_model_v3.pkl', 'rb') as f:
    v3 = pickle.load(f)

# v3's compute_5dim_boost (from train_5dim_v3.py)
def _cdb_v3(feats, p95, p99, feat_list):
    raw = 0.0
    for fname, baseline, coeff in feat_list:
        val = feats.get(fname, 0)
        thresh = max(p95.get(fname, 0), baseline)
        if val <= thresh: continue
        excess = val / thresh - 1.0
        contrib = coeff * (excess ** 0.6)
        if val > max(p99.get(fname, 0), baseline):
            p99_excess = val / max(p99.get(fname, 0), baseline) - 1.0
            p99_bonus = coeff * max(0, p99_excess) ** 0.6 * 0.5
            contrib += p99_bonus
        raw += contrib
    return raw

def cb_v3(fe):
    tn = max(fe.get('total_notes', 1), 1)
    d1 = _cdb_v3(fe, v3['p95_vals'], v3['p99_vals'], [
        ('tap_micro_max_0.0625beat', 2.0, 0.55), ('tap_micro_top5_0.0625beat', 1.2, 0.40),
        ('tap_burst_top5', 6.0, 0.35), ('jack_count', 20.0, 0.30),
        ('tap_per_second', 4.2, 0.30), ('very_short_interval_ratio', 0.18, 0.25),
        ('tap_burst_05_top5', 4.0, 0.35),
    ])
    mf3 = fe.get('multi_finger_3plus_events', 0)
    sm = fe.get('sim_pos_spread_max', 0)
    fmi = mf3 * sm / max(tn, 1) * 10
    d2 = 0.0
    th = max(v3['p99_vals'].get('multi_finger_3plus_events', 30), 1) * max(v3['p99_vals'].get('sim_pos_spread_max', 0.8), 0.1) / max(v3['p99_vals'].get('total_notes', 500), 1) * 10
    if fmi > max(th * 0.5, 0.3):
        exc = fmi / max(th * 0.5, 0.3) - 1
        d2 = 0.50 * (exc ** 0.6)
    d3 = _cdb_v3(fe, v3['p95_vals'], v3['p99_vals'], [
        ('wide_jump_count', 60.0, 0.40), ('burst_avg_movement', 1.5, 0.30),
        ('hold_lock_displacement_per_sec', 0.8, 0.40), ('movement_per_second', 7.0, 0.12),
        ('hold_lock_tap_events_per_hold', 1.0, 0.25),
    ])
    d4 = _cdb_v3(fe, v3['p95_vals'], v3['p99_vals'], [
        ('total_notes', 800.0, 0.45), ('tap_notes_per_second', 5.0, 0.35),
        ('notes_per_second', 7.5, 0.15), ('high_density_duration_ratio_16beat', 0.15, 0.20),
        ('sustained_density_run_count', 1.0, 0.18),
    ])
    d5 = _cdb_v3(fe, v3['p95_vals'], v3['p99_vals'], [
        ('density_transition_max', 2.5, 0.75), ('tempo_change_count', 30.0, 0.55),
        ('speed_change_total_impact', 20000, 0.28), ('offbeat_ratio', 0.08, 0.30),
        ('rhythm_entropy', 3.0, 0.15), ('bpm_change_count', 0.5, 0.30),
        ('density_transition_mean', 0.30, 0.38), ('type_switch_ratio', 0.06, 0.22),
        ('type_switch_per_sec', 0.8, 0.18),
    ])
    tb = d1 * 0.22 + d2 * 0.10 + d3 * 0.18 + d4 * 0.18 + d5 * 0.30
    tb = min(tb, 3.0)
    return tb, {'d1': d1, 'd2': d2, 'd3': d3, 'd4': d4, 'd5': d5}

charts = [
    ('DA\'AT', TD, '2155734445357448.json', 18.2),
    ('WakingShadows', TD, '93562988.json', 17.8),
    ('Chart_SP #13', TD, 'Chart_SP #1347(1).json', 17.6),
    ('105秒伝說', TD, 'Sigma (Haocore Mix) ~ 105秒の伝說 ~.json', 16.1),
    ('LiFE Garden(1.05x)', TD, '6923526264684294.json', 17.9),
    ('Far Eastern Flavor', DL, '61901444.json', 17.5),
    ('密码的周一', DL, '0582581966828779.json', 17.4),
    ('People people', DL, '1770391855.json', None),
    ('Galaxy Collapse', DL, '7009367902368871.json', None),
    ('Apollo', DL, 'Apollo(18.0).json', 18.0),
    ('Love & Justice', DL, 'Love & Justice(16.7)(1).json', 16.7),
    ('Xaleid◆scopiX', DL, 'Xaleid◆scopiX(18.2)(1).json', 18.2),
    ('silly-willy-nilly', DL, 'silly-willy-nilly(17.9)(1).json', 17.9),
    ('おぎゃりないざー', DL, 'おぎゃりないざー(16.5~16.6).json', 16.55),
    ('恋ひ恋ふ縁', DL, '恋ひ恋ふ縁(16.8)(1).json', 16.8),
    ('朧月', DL, '朧月(18.4)(1).json', 18.4),
    ('天方地園', DL, '天方地園(16.9)(1).json', 16.9),
    ('ニャンだふる♡サマー!!', DL, 'ニャンだふる♡サマー!!(15.8).json', 15.8),
    ('666', DL, '666(16.5).json', 16.5),
    ("Angel's Salad", DL, "Angel's Salad(16.9).json", 16.9),
    ('Breakcore革命前夜', DL, 'Breakcore革命前夜(16.3~16.5).json', 16.4),
    ('Cheerio!', DL, 'Cheerio!(17.1).json', 17.1),
    ('Lemegeton', DL, 'Lemegeton -little key of solomon-(16.6).json', 16.6),
    ('Submerged City', DL, 'Submerged City(18.0).json', 18.0),
]

print(f'{"谱面":<24} {"预期":>5} {"v3_GB":>7} {"v3_Boost":>7} {"v3_预测":>7} {"v3_误差":>8}  {"诊断":<16}')
print('-'*80)
for name, basedir, fname, exp in charts:
    fp = os.path.join(basedir, fname)
    try:
        cd = load_chart(fp)
        fe = extract_features(cd)
        # v3 prediction
        x = np.array([[fe.get(n, 0) for n in v3['feature_names']]])
        xs = v3['scaler'].transform(x)
        gb_v3 = float(v3['gb'].predict(xs)[0])
        bv3, ds = cb_v3(fe)
        pv3 = gb_v3 + bv3
        err3 = ''
        mk3 = ''
        if exp is not None:
            err3 = pv3 - exp
            mk3 = '' if abs(err3) < 0.31 else '⚠️' if abs(err3) < 0.5 else '🔴'
        # diagnostics
        diag = f'd1={ds["d1"]:.2f} d5={ds["d5"]:.2f}'
        es = f'{exp:.2f}' if exp else '?'
        print(f'{name:<24} {es:>5} {gb_v3:>7.3f} {bv3:>7.3f} {pv3:>7.3f} {err3:>+.3f} {mk3}  {diag}')
    except Exception as e:
        print(f'{name:<24}  ERROR: {e}')
