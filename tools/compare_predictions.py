# -*- coding: utf-8 -*-
"""批量对比：官谱（已知定数）vs 自制谱（Downloads 标注定数）的预测偏差"""
import os, sys, json, pickle
import numpy as np

ROOT = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator'
sys.path.insert(0, ROOT)
from feature_extractor import extract_features
from unified_parser import load_chart_from_bytes

# 模型
with open(os.path.join(ROOT, 'models', '6dim_model_v7.pkl'), 'rb') as f:
    m = pickle.load(f)
gb = m['gb']; scaler = m['scaler']
FN = m['feature_names']; P95 = m['p95_vals']; P99 = m['p99_vals']

# 复用 app.compute_boost
import importlib.util
spec = importlib.util.spec_from_file_location('appmod', os.path.join(ROOT, 'app.py'))
appmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(appmod)

def predict(feats):
    if not feats:
        return None
    x = np.array([[feats.get(n, 0) for n in FN]])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    p_boost, dims, _ = appmod.compute_boost(feats)
    return p_gb + p_boost, p_gb, p_boost, dims['categories'], feats

def load_and_predict(path):
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        cd, pe = load_chart_from_bytes(raw)
    except Exception as e:
        return None, f'解析失败: {e}'
    feats = extract_features(cd)
    r = predict(feats)
    if r is None:
        return None, '特征提取失败'
    return r, None

# ===== 官谱 =====
from data_loader import load_difficulty_tsv, find_chart_files
tsv = os.path.join(ROOT, 'data', 'info', 'difficulty.tsv')
diffs = load_difficulty_tsv(tsv)
chart_files = find_chart_files(os.path.join(ROOT, 'data', 'chart'))

print('=' * 78)
print('【官谱】真实定数 vs 预测')
print('=' * 78)
official = []
for folder, info in list(chart_files.items()):
    sid = info['song_id']
    if sid not in diffs:
        continue
    for lv, path in info['levels'].items():
        if lv not in diffs[sid]:
            continue
        with open(path, 'rb') as f:
            raw = f.read()
        try:
            cd, pe = load_chart_from_bytes(raw)
            feats = extract_features(cd)
            if not feats:
                continue
            r = predict(feats)
            real = diffs[sid][lv]
            official.append((real, r[0], r[1], r[2], folder, lv, feats))
        except Exception as e:
            pass

official.sort(key=lambda x: -x[0])
print(f'共 {len(official)} 个官谱')
errs = [p - r for r, p, _, _, _, _, _ in official]
print(f'平均误差 (预测-真实): {np.mean(errs):+.3f}  中位: {np.median(errs):+.3f}')
for real, pred, gb_, boost_, folder, lv, feats in official[:12]:
    print(f'  {folder} [{lv}] 真实={real:6.2f}  预测={pred:6.2f}  (GB={gb_:.2f}+Boost={boost_:.2f})  d={pred-real:+.2f}')

print()
print('=' * 78)
print('【自制谱】Downloads 标注定数 vs 预测')
print('=' * 78)
DL = r'C:\Users\NaNK\Downloads'
charts = {
    'Apollo(17.8).json': 17.8,
    'Submerged City(17.8).json': 17.8,
    'The Final EndGame(18.4).json': 18.4,
    'Runengon(16.2~16.4).json': 16.3,
    'Breakcore革命前夜(16.5~16.7).json': 16.6,
    'おぎゃりないざー(16.4~16.6).json': 16.5,
    'ギザバ怪文書(18.3).json': 18.3,
    'シグマ(16.4~16.6).json': 16.5,
    'Cheerio!(17.0).json': 17.0,
    'Xaleid◆scopiX(18.2)(1).json': 18.2,
    'silly-willy-nilly(17.7)(1).json': 17.7,
    'トキラキメキ(14.6)(1).json': 14.6,
    '恋ひ恋ふ縁(16.8)(1).json': 16.8,
    '茉子の日常(15.5).json': 15.5,
    'ニャンだふる♡サマー!!(15.8).json': 15.8,
    'ふたりのスタートボタン(13.4).json': 13.4,
    'スタートリップ(12.2).json': 12.2,
}
# 追加带定数的自制谱文件
for fn in os.listdir(DL):
    if fn.endswith('.json'):
        base = fn.rsplit('.', 1)[0]
        # 提取括号中的数字
        import re
        mm = re.search(r'\((\d+(?:\.\d+)?)(?:~(\d+(?:\.\d+)?))?\)', base)
        if mm:
            v = float(mm.group(1))
            charts.setdefault(fn, v)

for fn, real in sorted(charts.items(), key=lambda x: -x[1]):
    path = os.path.join(DL, fn)
    if not os.path.exists(path):
        continue
    r, err = load_and_predict(path)
    if r is None:
        print(f'  {fn}  真实={real:6.2f}  {err}')
        continue
    pred, gb_, boost_, cats, feats = r
    print(f'  {fn[:30]:30s} 真实={real:6.2f}  预测={pred:6.2f}  (GB={gb_:.2f}+B={boost_:.2f})  d={pred-real:+.2f}  n={feats.get("total_notes"):5d} dur={feats.get("duration_sec"):6.1f}s nps={feats.get("real_core_notes_per_second"):.2f}')
