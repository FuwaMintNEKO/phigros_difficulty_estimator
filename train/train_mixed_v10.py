# -*- coding: utf-8 -*-
"""混合训练实验: 官谱(抽大部分) + 上架谱 + 未上架谱 混合训练

架构与 train_final_v10.py 一致: GB残差(250特征) + boost叠加(48手动特征)
数据源:
  1. 官谱 data/chart (官方定数 difficulty.tsv)
  2. 上架 data/phira/json (社区定数 charts.json)
  3. 未上架 data/phira/json_unranked (社区定数 unranked_final_download.json)

--mix_official 官谱抽取比例 (默认 1.0 = 全部)
--mix_community 社区谱抽取比例 (默认 0.5)
--lvmerge 1=3类EZ/HD/IN_AT, 0=4类
输出: models/6dim_model_v10_mixed.pkl + CV报告
"""
import os, sys, io, pickle, argparse
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from feature_extractor import extract_features
from unified_parser import load_chart_from_bytes
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from boost_config import MANUAL_FLAT

parser = argparse.ArgumentParser()
parser.add_argument('--variant', default='V14', choices=['V14', 'V15'])
parser.add_argument('--caps', type=float, default=4.0)
parser.add_argument('--lowweight', type=float, default=1.5)
parser.add_argument('--lvmerge', type=int, default=1)
parser.add_argument('--mix_official', type=float, default=1.0, help='官谱抽取比例')
parser.add_argument('--mix_community', type=float, default=0.5, help='社区谱抽取比例')
parser.add_argument('--calib', type=int, default=0, help='1=社区定数标签按段校准(官方口径)')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--out', default='6dim_model_v10_mixed.pkl')
args = parser.parse_args()

OVERRIDE = {'V14': {}, 'V15': {'type_switch_per_sec': 0.06, 'chord_alternation_rate': 0.15,
            'weighted_mf_score_per_sec': 0.20, 'above_avg_duration_sec': 0.42}}[args.variant]
FLAT = []
d = {f: (bl, co) for f, bl, co in MANUAL_FLAT}
for f, new_co in OVERRIDE.items():
    d[f] = (d[f][0], new_co)
FLAT = [(f, bl, co) for f, (bl, co) in d.items()]
CAPS = {'_default': args.caps} if args.caps else {}
LV_ORDER = ['EZ', 'HD', 'IN', 'AT']
if args.lvmerge:
    LV_ORDER = ['EZ', 'HD', 'IN_AT']
# 社区定数 → 官方口径 校准表 (median bias of pred-diff, 来自 3类生产模型预测:
#  上架谱: >=16.5 +0.012, 14-16.5 +0.277, 11-14 +0.169
#  未上架: >=16.5 +0.186, 14-16.5 +0.147
# 校准标签 = 社区定数 - 校准量)
CALIB = {
    ('rkd', 16.5): 0.012, ('rkd', 14.0): 0.277, ('rkd', 11.0): 0.169,
    ('unr', 16.5): 0.186, ('unr', 14.0): 0.147,
}

def calib_amount(src, diff):
    if not args.calib:
        return 0.0
    # 按段下限匹配
    if diff >= 16.5:
        return CALIB.get((src, 16.5), 0.0)
    if diff >= 14.0:
        return CALIB.get((src, 14.0), 0.0)
    if diff >= 11.0:
        return CALIB.get((src, 11.0), 0.0)
    return 0.0
print(f'混合训练: 官谱={args.mix_official}, 社区={args.mix_community}, lvmerge={args.lvmerge}, 校准={bool(args.calib)}')

# ===== 加载社区定数 =====
def parse_level(lv_str):
    """社区谱 level 五花八门: 提取 EZ/HD/IN/AT 前缀, 特殊前缀归 IN"""
    s = str(lv_str).strip().upper().replace(' ', '')
    for lv in ['AT', 'IN', 'HD', 'EZ']:
        if s.startswith(lv):
            return lv
    return None

charts = json_ = None
import json as _json
meta_ranked = {}
for lst in _json.load(open(os.path.join(_ROOT, 'data', 'phira', 'charts.json'), encoding='utf-8')).values():
    for c in lst:
        meta_ranked[c['id']] = c
meta_unranked = {}
for c in _json.load(open(os.path.join(_ROOT, 'data', 'phira', 'unranked_final_download.json'), encoding='utf-8')):
    meta_unranked[c['id']] = c

# ===== 1. 官谱 =====
CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
song_difficulties = load_difficulty_tsv(os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv'))
chart_files = find_chart_files(CHART_DIR)
off_items = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties:
        continue
    diffs = song_difficulties[sid]
    for lv in ['EZ', 'HD', 'IN', 'AT']:
        if lv in info['levels'] and lv in diffs:
            off_items.append({'path': info['levels'][lv], 'diff': diffs[lv], 'lv': lv, 'group': fn, 'src': 'off'})

# ===== 2. 上架谱 =====
J_DIR = os.path.join(_ROOT, 'data', 'phira', 'json')
rkd_items = []
for fn in os.listdir(J_DIR):
    if not fn.endswith('.json'):
        continue
    cid = int(fn[:-5])
    c = meta_ranked.get(cid)
    if not c:
        continue
    lv = parse_level(c.get('level', ''))
    if lv is None:
        lv = 'IN'
    d = c.get('difficulty', 0)
    d_c = d - calib_amount('rkd', d)
    rkd_items.append({'path': os.path.join(J_DIR, fn), 'diff': d_c,
                      'diff_orig': d, 'lv': lv, 'group': c.get('name', f'r{cid}'), 'src': 'rkd'})

# ===== 3. 未上架谱 =====
U_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked')
unr_items = []
for fn in os.listdir(U_DIR):
    if not fn.endswith('.json'):
        continue
    cid = int(fn[:-5])
    c = meta_unranked.get(cid)
    if not c:
        continue
    lv = parse_level(c.get('level', ''))
    if lv is None:
        lv = 'IN'
    d = c.get('difficulty', 0)
    d_c = d - calib_amount('unr', d)
    unr_items.append({'path': os.path.join(U_DIR, fn), 'diff': d_c,
                      'diff_orig': d, 'lv': lv, 'group': c.get('name', f'u{cid}'), 'src': 'unr'})

print(f'官谱 {len(off_items)}, 上架 {len(rkd_items)}, 未上架 {len(unr_items)}')

# ===== 抽取 =====
rng = np.random.RandomState(args.seed)
def sample(items, ratio):
    if ratio >= 1.0:
        return list(items)
    n = int(len(items) * ratio)
    idx = rng.choice(len(items), n, replace=False)
    return [items[i] for i in idx]

off_use = sample(off_items, args.mix_official)
com_use = sample(rkd_items + unr_items, args.mix_community)
print(f'抽取后: 官谱 {len(off_use)}, 社区 {len(com_use)}')

# ===== 特征提取 =====
feats_list, labels, labels_orig, levels_list, groups_list, src_list = [], [], [], [], [], []
fails = []
for item in off_use:
    try:
        feats = extract_features(load_chart_json(item['path']))
        if feats:
            feats_list.append(feats); labels.append(item['diff'])
            labels_orig.append(item.get('diff_orig', item['diff']))
            levels_list.append(item['lv']); groups_list.append(item['group']); src_list.append('off')
    except Exception:
        pass
for item in com_use:
    try:
        with open(item['path'], 'rb') as f:
            raw = f.read()
        cd, _ = load_chart_from_bytes(raw)
        feats = extract_features(cd, speed=1.0)
        if feats:
            feats_list.append(feats); labels.append(item['diff'])
            labels_orig.append(item.get('diff_orig', item['diff']))
            levels_list.append(item['lv']); groups_list.append(item['group']); src_list.append(item['src'])
    except Exception as e:
        fails.append((item['path'], str(e)[:40]))
print(f'特征提取成功: {len(feats_list)}, 失败: {len(fails)}')
for p, e in fails[:8]:
    print(f'  FAIL {os.path.basename(p)}: {e}')

n = len(feats_list)
feature_names = sorted(feats_list[0].keys())
GB_EXCLUDE_KEYWORDS = [
    'stop_go', 'track_section', 'offbeat_ratio', 'dense_mf',
    'mf_burst', 'mf_events_per_second', 'mf_with_hold',
    'cross_line_3plus', 'min_interval_beats',
    'multi_finger_3plus', 'multi_finger_4plus', 'multi_finger_max',
    'chord_size_entropy', 'chord_3note', 'chord_4plus',
    'long_jack', 'short_jack', 'jack_max_run',
    'per_second', 'per_sec', 'rate_per_sec',
    'total_movement', 'total_steps', 'total_event',
    'total_hold_duration', 'total_chord',
    'speed_change_total',
    'micro_max_', 'micro_spike_',
    'density_above_zero', 'core_density_above_zero',
    'density_skew', 'density_transition_max',
    'avg_hold_duration', 'max_hold_duration',
    'finger_vs_total',
    'note_speed', 'flash_', 'visible_time', 'chord_jack', 'fast_hold',
]
GB_KEEP = {'density_dimension', 'real_core_notes_per_second',
           'core_peak_density_1sec_top5avg', 'core_peak_density_top5avg_1beat',
           'movement_per_second', 'movement_density_index'}
gb_feature_names = [nn for nn in feature_names
                    if nn in GB_KEEP or not any(kw in nn for kw in GB_EXCLUDE_KEYWORDS)]
print(f'GB特征数: {len(gb_feature_names)}')

X_base = np.array([[f.get(nn, 0) for nn in gb_feature_names] for f in feats_list])
y = np.array(labels)
X_lv = np.zeros((n, len(LV_ORDER)))
for i, lv in enumerate(levels_list):
    key = 'IN_AT' if (args.lvmerge and lv in ('IN', 'AT')) else lv
    if key not in LV_ORDER:
        key = 'IN_AT' if 'IN_AT' in LV_ORDER else 'IN'
    X_lv[i, LV_ORDER.index(key)] = 1.0
levels_arr = np.array(levels_list)
SAMPLE_W = np.ones(n)
if args.lowweight != 1.0:
    SAMPLE_W[levels_arr == 'EZ'] = args.lowweight
    SAMPLE_W[levels_arr == 'HD'] = args.lowweight

def compute_boost_v9(feats, p95_vals, p99_vals):
    total = 0.0
    cap = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0)
        pv = p95_vals.get(fname, 0)
        t = max(pv * 0.55, bl * 0.5)
        if v <= t:
            continue
        e = v / t - 1.0
        c = CAPS.get(fname, cap)
        if c is not None and e > c:
            e = c
        x = co * (e ** 0.70)
        if v > max(p99_vals.get(fname, 0), bl * 0.5):
            pe = v / max(p99_vals.get(fname, 0), bl * 0.5) - 1.0
            if c is not None and pe > c:
                pe = c
            x += co * max(0, pe) ** 0.70 * 0.5
        total += x
    return total

# ===== 歌曲分组 5 折 CV =====
N_FOLDS = 5
gkf = GroupKFold(n_splits=N_FOLDS)
splits = list(gkf.split(X_base, y, groups=np.array(groups_list)))
oof = np.zeros(n)
for fi, (tr, te) in enumerate(splits):
    p95_vals, p99_vals = {}, {}
    for j, name in enumerate(gb_feature_names):
        col = X_base[tr, j]
        p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
        p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
    boosts = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in feats_list])
    X_tr = np.hstack([X_base[tr], X_lv[tr]])
    X_te = np.hstack([X_base[te], X_lv[te]])
    sc = StandardScaler().fit(X_tr)
    gb = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
    gb.fit(sc.transform(X_tr), y[tr] - boosts[tr], sample_weight=SAMPLE_W[tr])
    oof[te] = gb.predict(sc.transform(X_te)) + boosts[te]
    print(f'  fold{fi} OOF完成', flush=True)

src_arr = np.array(src_list)
y_orig = np.array(labels_orig)
print(f'\n===== 混合训练 歌曲分组5折CV (OOF) =====')
print(f'整体MAE = {mean_absolute_error(y, oof):.4f}  R2={r2_score(y, oof):.4f}  n={n}')
for src in ['off', 'rkd', 'unr']:
    m = np.where(src_arr == src)[0]
    if len(m):
        yy = y_orig[m] if src != 'off' else y[m]
        print(f'  {src}: n={len(m)} MAE={mean_absolute_error(yy, oof[m]):.4f} '
              f'偏差(对原始定数)={np.mean(oof[m] - yy):+.3f}')
for lv in LV_ORDER:
    m = np.where(levels_arr == lv)[0]
    if len(m):
        print(f'  {lv}: n={len(m)} MAE={mean_absolute_error(y[m], oof[m]):.4f}')
off_idx = np.where(src_arr == 'off')[0]
if len(off_idx):
    print(f'\n官谱子集(测试新谱泛化口径): n={len(off_idx)} MAE={mean_absolute_error(y[off_idx], oof[off_idx]):.4f}')
    # 官谱按难度段
    for lo, hi, label in [(16.5, 99, '>=16.5'), (14, 16.5, '14-16.5'), (11, 14, '11-14'), (0, 11, '<11')]:
        m = off_idx[(y[off_idx] >= lo) & (y[off_idx] < hi)]
        if len(m):
            print(f'    官谱 {label}: n={len(m)} MAE={mean_absolute_error(y[m], oof[m]):.4f} '
                  f'偏差={np.mean(oof[m] - y[m]):+.3f}')

# ===== 全量重训 =====
p95_vals, p99_vals = {}, {}
for j, name in enumerate(gb_feature_names):
    col = X_base[:, j]
    p95_vals[name] = float(np.percentile(col, 95)) if np.max(col) > 0 else 0
    p99_vals[name] = float(np.percentile(col, 99)) if np.max(col) > 0 else 0
boosts = np.array([compute_boost_v9(f, p95_vals, p99_vals) for f in feats_list])
X_all = np.hstack([X_base, X_lv])
scaler = StandardScaler().fit(X_all)
gb_final = GradientBoostingRegressor(n_estimators=500, max_depth=5, min_samples_leaf=3,
                                     learning_rate=0.05, subsample=0.8, random_state=42)
gb_final.fit(scaler.transform(X_all), y - boosts, sample_weight=SAMPLE_W)

model = {
    'gb': gb_final, 'scaler': scaler, 'feature_names': gb_feature_names,
    'p95_vals': p95_vals, 'p99_vals': p99_vals, 'lv_order': LV_ORDER,
    'version': f'10.1-mixed-{args.variant}-off{args.mix_official}-com{args.mix_community}-lvmerge{args.lvmerge}',
    'n_train': n, 'MANUAL_FLAT': FLAT, 'caps': CAPS,
    'train_meta': {'n': n, 'songs': len(set(groups_list)),
                   'cv_mae': float(mean_absolute_error(y, oof)), 'cv_r2': float(r2_score(y, oof)),
                   'src_counts': {s: int((src_arr == s).sum()) for s in ['off', 'rkd', 'unr']}},
}
path = os.path.join(_ROOT, 'models', args.out)
with open(path, 'wb') as f:
    pickle.dump(model, f)
print(f'\n已保存: {path}')
