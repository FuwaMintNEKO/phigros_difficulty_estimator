# -*- coding: utf-8 -*-
"""实验C: kyou分类器改进 — 更多特征 + 调参
"""
import os, sys, pickle, numpy as np, io, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
kyou = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'kyou_tags.json'), encoding='utf-8'))
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = cache['official']
def norm(s):
    return re.sub(r'[^a-z0-9\u4e00-\u9fff]', '', s.lower())
# 全部GB特征 (模型里有的)
m = pickle.load(open(os.path.join(_ROOT, 'models', '6dim_model_v11_7b.pkl'), 'rb'))
ALL_FEATS = list(m['feature_names'])
rows = []
for k in kyou:
    kn = norm(k['song'])
    for r in official:
        if r['level'] in ('IN', 'AT') and kn and kn in norm(r['name']):
            rows.append((k['tag'].replace('?', '').strip(), r['feats']))
            break
skf = StratifiedKFold(5, shuffle=True, random_state=42)
# 尝试: 全特征 vs 精选特征 vs 不同深度
configs = [
    ('全273特征 depth2', ALL_FEATS, 2, 300),
    ('全273特征 depth3', ALL_FEATS, 3, 200),
    ('全273特征 depth4', ALL_FEATS, 4, 200),
    ('精选19特征 depth3', ['above_avg_density_mean', 'eff_avg_tps_1s', 'weighted_mf_score_per_sec', 'stair_speed_avg',
                          'thirtysecond_run_ratio', 'fast_ms_100_ratio', 'jline_movement_density', 'tempo_change_log_density',
                          'above_avg_duration_sec', 'bpm', 'jack_density', 'chord_jack_3plus_pairs', 'movement_per_second',
                          'chord_events_peak_8s', 'avg_movement', 'position_iqr', 'rhythm_entropy', 'pattern_switch_rate', 'drag_ratio'], 3, 200),
    ('全特征+标签输入', None, 3, 200),
]
for tag, feats, depth, n_est in configs:
    if feats is None:
        # 特征+标签数特征
        TH = json.load(open(os.path.join(_ROOT, 'data', 'tag_thresholds.json'), encoding='utf-8'))
        X = []
        for c, f in rows:
            v = [f.get(k, 0) for k in ALL_FEATS]
            ntag = sum(1 for k, th in TH.items() if k != '定轨' and f.get(k, 0) >= th)
            v.append(ntag)
            X.append(v)
        X = np.array(X)
    else:
        X = np.array([[f.get(k, 0) for k in feats] for _, f in rows])
    y = np.array([c for c, _ in rows])
    clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=depth, learning_rate=0.06, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
    print(f'{tag:<22} CV={scores.mean():.3f} ± {scores.std():.3f}')
print('DONE')
