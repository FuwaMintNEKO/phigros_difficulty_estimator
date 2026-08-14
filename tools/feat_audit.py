# -*- coding: utf-8 -*-
"""系统审查特征计算: 各特征 vs 官谱定数的相关性 + Melodiniq类谱定位
目标: 找出 24分/爆发/高速 特征在Melodiniq上失真的环节"""
import os, sys, io, json, pickle, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from scipy.stats import spearmanr
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

# Melodiniq 全特征与 Verrückt 对比 (重点: 爆发/高速类)
p_mel = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p_mel, 'rb') as f:
    cd_mel, _ = load_chart_from_bytes(f.read())
f_mel = extract_features(cd_mel, speed=1.0)
p_ver = os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')
with open(p_ver, 'rb') as f:
    cd_ver, _ = load_chart_from_bytes(f.read())
f_ver = extract_features(cd_ver, speed=1.0)

# 爆发/速度/密度相关特征对比
KEYS = ['micro_max_0.0625beat','micro_max_0.125beat','micro_peak_top5_0.0625beat','micro_peak_top5_0.125beat',
        'tap_micro_top5_0.0625beat','tap_micro_top5_0.125beat','tap_burst_top5','tap_burst_05_top5',
        'miniburst_count','miniburst_density','global_jack_count','fast_ms_050_ratio','fast_ms_100_ratio',
        'fast_ms_150_ratio','real_notes_per_second','real_core_notes_per_second','above_avg_density_mean',
        'eff_peak_tps_1s','eff_avg_tps_1s','above_avg_duration_sec','movement_per_second','movement_density_index',
        'cross_hand_density','lane_switch_density','speed_volatility','tempo_change_log_density','rhythm_entropy']
print(f'{"特征":<34}{"Melodiniq":>10}{"Verrückt":>10}{"Mel排名(官谱)":>12}')
# ranked 官谱分布
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
official = cache['official']
for k in KEYS:
    vm = f_mel.get(k, 0); vv = f_ver.get(k, 0)
    vals = np.array([o['feats'].get(k, 0) for o in official])
    pct = np.mean(vals <= vm)*100 if vals.max() > 0 else 0
    print(f'{k:<34}{vm:>10.2f}{vv:>10.2f}{pct:>11.0f}%')
print('DONE')