# -*- coding: utf-8 -*-
"""量化两个疑点对预测的影响:
A. feature_extractor HEAD版 vs 当前版 的特征漂移 (wide_jump阈值1.5→2.5) 对GB预测的影响
B. RPE坐标 ÷100(当前) vs ÷75(正确) 对预测的影响
"""
import os, sys, json, pickle, numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
import feature_extractor as fe_current
import _feat_git_head as fe_head

with open(os.path.join(_ROOT, 'models', '6dim_model_v7.pkl'), 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']; FN = m['feature_names']

def gb_predict(feats):
    x = np.array([[feats.get(n, 0) for n in FN]])
    return float(gb.predict(scaler.transform(x))[0])

# ====== A. 特征漂移 ======
print('='*70)
print('A. HEAD vs 当前 feature_extractor 漂移')
print('='*70)
CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
DIFFICULTY_TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
song_difficulties = load_difficulty_tsv(DIFFICULTY_TSV)
chart_files = find_chart_files(CHART_DIR)

diffs = []
drift_songs = 0
n = 0
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in song_difficulties: continue
    for lv in ['EZ','HD','IN','AT']:
        if lv not in info['levels']: continue
        n += 1
        try:
            cd = load_chart_json(info['levels'][lv])
            f1 = fe_current.extract_features(cd)
            f2 = fe_head.extract_features(cd)
            if not f1 or not f2: continue
            # 比较所有219个GB特征
            drift = 0
            changed = []
            for fn_ in FN:
                a = f1.get(fn_, 0); b = f2.get(fn_, 0)
                if abs(a - b) > 1e-9:
                    drift += 1
                    changed.append((fn_, b, a))
            if drift > 0:
                drift_songs += 1
            p1 = gb_predict(f1); p2 = gb_predict(f2)
            if abs(p1-p2) > 0.01:
                diffs.append((fn, lv, p2, p1, drift, changed))
        except Exception as e:
            pass
print(f'共{n}个谱面, 特征漂移的谱面数: {drift_songs}')
print(f'GB预测变化>0.01的谱面数: {len(diffs)}')
if diffs:
    changes = [abs(d[2]-d[3]) for d in diffs]
    print(f'GB预测变化量: 最大={max(changes):.3f}, 均值={np.mean(changes):.3f}')
    for fn, lv, p2, p1, drift, changed in diffs[:8]:
        print(f'  {fn:<35} {lv}: HEAD_GB={p2:.3f} -> CUR_GB={p1:.3f} (delta={p1-p2:+.3f}), 变化特征={drift}个')
        for cf, ov, nv in changed[:5]:
            print(f'      {cf}: {ov:.3f} -> {nv:.3f}')
else:
    print('无显著GB预测差异 (漂移可忽略)')

# ====== B. RPE坐标缩放影响 ======
print()
print('='*70)
print('B. RPE坐标 ÷100(当前) vs ÷75(正确) 预测影响')
print('='*70)
import predict_rpe
DL = r'C:\Users\NaNK\Downloads'
for fname in ['Apollo(17.8).json', 'ギザバ怪文書(18.3).json', 'ふたりのスタートボタン(13.4).json']:
    path = os.path.join(DL, fname)
    if not os.path.exists(path):
        continue
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    # 当前: ÷100
    chart_cur = predict_rpe.convert_rpe_to_standard(raw)
    # 正确: ÷75 — 直接修改转换后的positionX
    chart_fix = json.loads(json.dumps(chart_cur))
    for line in chart_fix['judgeLineList']:
        for n_ in line.get('notesAbove', []) + line.get('notesBelow', []):
            n_['positionX'] = n_['positionX'] * 100.0 / 75.0
    f_cur = fe_current.extract_features(chart_cur)
    f_fix = fe_current.extract_features(chart_fix)
    if not f_cur or not f_fix:
        continue
    p_cur = gb_predict(f_cur)
    p_fix = gb_predict(f_fix)
    # boost也要看(用v9近似不需要, 只看GB + 关键特征)
    print(f'{fname}:')
    print(f'  position_std: {f_cur.get("position_std",0):.3f} -> {f_fix.get("position_std",0):.3f}')
    print(f'  avg_movement: {f_cur.get("avg_movement",0):.3f} -> {f_fix.get("avg_movement",0):.3f}')
    print(f'  movement_per_second: {f_cur.get("movement_per_second",0):.3f} -> {f_fix.get("movement_per_second",0):.3f}')
    print(f'  cross_hand_density: {f_cur.get("cross_hand_density",0):.3f} -> {f_fix.get("cross_hand_density",0):.3f}')
    print(f'  wide_jump_density: {f_cur.get("wide_jump_density",0):.3f} -> {f_fix.get("wide_jump_density",0):.3f}')
    print(f'  GB: {p_cur:.3f} -> {p_fix:.3f} (delta={p_fix-p_cur:+.3f})')
    print()
