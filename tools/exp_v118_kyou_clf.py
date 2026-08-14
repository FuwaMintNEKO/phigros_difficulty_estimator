# -*- coding: utf-8 -*-
"""训练kyou分类器: 特征 → 6类 (GB分类, 299官谱样本)
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
FEATS = ['above_avg_density_mean', 'eff_avg_tps_1s', 'weighted_mf_score_per_sec', 'stair_speed_avg',
         'thirtysecond_run_ratio', 'fast_ms_100_ratio', 'jline_movement_density', 'tempo_change_log_density',
         'above_avg_duration_sec', 'bpm', 'jack_density', 'chord_jack_3plus_pairs', 'movement_per_second',
         'chord_events_peak_8s', 'avg_movement', 'position_iqr', 'rhythm_entropy', 'pattern_switch_rate', 'drag_ratio']
rows = []
for k in kyou:
    kn = norm(k['song'])
    for r in official:
        if r['level'] in ('IN', 'AT') and kn and kn in norm(r['name']):
            rows.append((k['tag'], r['feats']))
            break
X = np.array([[f.get(k, 0) for k in FEATS] for _, f in rows])
y = np.array([c for c, _ in rows])
print(f'样本: {len(rows)}, 类别: {dict(zip(*np.unique(y, return_counts=True)))}')
clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
print(f'交叉验证准确率: {scores.mean():.3f} ± {scores.std():.3f}')
# 各类准确率
from sklearn.model_selection import cross_val_predict
pred = cross_val_predict(clf, X, y, cv=skf)
from collections import Counter
for cat in ['硬抗', '综合', '定位', '读谱', '拆谱', '多指']:
    mk = y == cat
    if mk.sum():
        print(f'  {cat}: n={mk.sum()} 正确={np.sum(pred[mk]==cat)} ({100*np.sum(pred[mk]==cat)/mk.sum():.0f}%)')
# 全量训练保存
clf.fit(X, y)
import pickle as pk
pk.dump({'clf': clf, 'feats': FEATS, 'classes': list(clf.classes_)}, open(os.path.join(_ROOT, 'models', 'kyou_classifier.pkl'), 'wb'))
print('已保存: models/kyou_classifier.pkl')
print('DONE')
