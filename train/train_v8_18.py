"""v8.18: 极端谱面诊断 — 分析为什么某些谱面被低估"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os, pickle, numpy as np, math, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')

print("=" * 60)
print("  v8.18 — 极端谱面诊断")
print("=" * 60)

song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)

all_charts = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    for lv in ['IN', 'AT']:
        if lv not in info.get('levels', {}): continue
        if lv not in song_difficulties[sid]: continue
        try:
            cd = load_chart_json(info['levels'][lv])
            feats = extract_features(cd)
            if feats:
                feats['_difficulty'] = song_difficulties[sid][lv]
                feats['_name'] = fn[:30]
                all_charts.append(feats)
        except Exception as e: pass

exclude_patterns = ['snowmelt', 'snowdance', 'snow dance']
all_charts = [f for f in all_charts if not any(p.lower() in f['_name'].lower() for p in exclude_patterns)]
print(f'总谱面数: {len(all_charts)}')

FNo = sorted({k for f in all_charts for k in f.keys() if not k.startswith('_')})

diffs = np.array([f['_difficulty'] for f in all_charts])
bins = np.digitize(diffs, bins=[13, 14, 15, 16, 17])
train_mask = np.zeros(len(all_charts), dtype=bool)
test_mask = np.zeros(len(all_charts), dtype=bool)
np.random.seed(42)
for b in range(1, 6):
    idx = np.where(bins == b)[0]
    if len(idx) == 0: continue
    tr_idx, te_idx = train_test_split(idx, test_size=0.25, random_state=42)
    train_mask[tr_idx] = True
    test_mask[te_idx] = True

train_charts = [all_charts[i] for i in range(len(all_charts)) if train_mask[i]]
test_charts = [all_charts[i] for i in range(len(all_charts)) if test_mask[i]]

train_targets = np.array([f['_difficulty'] for f in train_charts])
test_targets = np.array([f['_difficulty'] for f in test_charts])
X_train = np.array([[f.get(n, 0) for n in FNo] for f in train_charts])
X_test = np.array([[f.get(n, 0) for n in FNo] for f in test_charts])

# 全数据训练
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([train_targets, test_targets])
all_names = [f['_name'] for f in train_charts] + [f['_name'] for f in test_charts]

xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=1)
xgb_model.fit(X_all, y_all)

# 全数据预测和残差
y_pred_all = xgb_model.predict(X_all)
residuals = y_all - y_pred_all

# 找出最被低估的谱面（残差最大）
underest_idx = np.argsort(residuals)[-20:]  # 正残差 = 真实值 > 预测值（低估）

print("\n===== Top 20 最被低估的谱面 =====")
print(f'{"Name":<35s} {"True":>6s} {"Pred":>8s} {"Residual":>9s}')
underestimated = []
for i in underest_idx:
    name = all_names[i][:35]
    true = y_all[i]
    pred = y_pred_all[i]
    res = residuals[i]
    underestimated.append(i)
    print(f'{name:<35s} {true:>6.1f} {pred:>8.2f} {res:>+9.2f}')

# 分析这些谱面的共同特征
print("\n===== 低估谱面 vs 正常谱面: 特征差异分析 =====")
# 找与低估谱面同难度区间的正常谱面
underest_diff = y_all[underestimated]
normal_mask = np.ones(len(y_all), dtype=bool)
normal_mask[underestimated] = False
# 正常谱面中，选同难度区间
normal_in_range = []
for i in underestimated:
    diff = y_all[i]
    candidates = np.where((normal_mask) & (np.abs(y_all - diff) < 0.5))[0]
    if len(candidates) > 0:
        normal_in_range.extend(candidates[:5])

normal_in_range = list(set(normal_in_range))
print(f'低估谱面: {len(underestimated)} 个, 正常谱面: {len(normal_in_range)} 个')

# 对比关键特征
key_features = [
    'real_notes_per_second', 'peak_tps_1sec', 'notes_per_second',
    'core_peak_density', 'above_avg_density_mean', 'peak_density_1sec',
    'jack_count', 'trill_count', 'hold_ratio', 'drag_ratio',
    'click_ratio', 'flick_ratio', 'bpm_mean', 'bpm_max',
    'note_count', 'duration_seconds', 'max_hold_duration',
    'speed_change_count', 'bpm_volatility'
]

print(f'\n{"Feature":<25s} {"低估均值":>10s} {"正常均值":>10s} {"差":>8s} {"差/正常%":>10s}')
for feat in key_features:
    if feat not in FNo: continue
    fi = FNo.index(feat)
    u_vals = X_all[underestimated, fi]
    n_vals = X_all[normal_in_range, fi]
    u_mean = np.mean(u_vals)
    n_mean = np.mean(n_vals)
    diff = u_mean - n_mean
    pct = (diff / n_mean * 100) if n_mean != 0 else 0
    print(f'{feat:<25s} {u_mean:>10.4f} {n_mean:>10.4f} {diff:>+8.4f} {pct:>+9.1f}%')

# 打印每个低估谱面的关键特征值
print("\n===== 每个低估谱面的关键特征 =====")
print(f'{"Name":<35s} {"r_nps":>7s} {"p_tps":>7s} {"jack":>6s} {"trill":>6s} {"hold%":>6s} {"drag%":>6s} {"bpm_vol":>8s}')
for i in underestimated:
    name = all_names[i][:35]
    vals = {}
    for feat in ['real_notes_per_second', 'peak_tps_1sec', 'jack_count', 'trill_count', 'hold_ratio', 'drag_ratio', 'bpm_volatility']:
        if feat in FNo:
            vals[feat] = X_all[i, FNo.index(feat)]
        else:
            vals[feat] = 0
    print(f'{name:<35s} {vals["real_notes_per_second"]:>7.2f} {vals["peak_tps_1sec"]:>7.2f} {vals["jack_count"]:>6.0f} {vals["trill_count"]:>6.0f} {vals["hold_ratio"]:>6.3f} {vals["drag_ratio"]:>6.3f} {vals["bpm_volatility"]:>8.4f}')

print("\n===== 完成 =====")