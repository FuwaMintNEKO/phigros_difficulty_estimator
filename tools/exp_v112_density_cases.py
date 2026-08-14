# -*- coding: utf-8 -*-
"""t2 问题谱例验证: ギザバ怪文書 + Sigma Regrets"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
_ROOT = r'D:\\Trae项目\\新建文件夹\\phigros_difficulty_estimator'
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features
from tools.exp_v112_density_planB import eff_density_features

cases = [
    (r'D:\\Trae项目\\新建文件夹\\phigros_difficulty_estimator\\data\\test_charts\\ギザバ怪文書(18.3).json', 'ギザバ怪文書(18.3)'),
    (r'D:\\Trae项目\\新建文件夹\\phigros_difficulty_estimator\\data\\chart\\used_test_data\\Sigma (Haocore Mix) ~ Regrets of The Yellow Tuli.json', 'Sigma Regrets'),
]
for path, label in cases:
    print(f'===== {label} =====')
    try:
        with open(path, 'rb') as f:
            cd, raw = load_chart_from_bytes(f.read())
        feats = extract_features(cd, speed=1.0)
        if not feats:
            print('  extract_features 返回 None')
            continue
        dens = feats.get('above_avg_density_mean', 0)
        effa = feats.get('eff_avg_tps_1s', 0)
        effp = feats.get('eff_peak_tps_1s', 0)
        rcnps = feats.get('real_core_notes_per_second', 0)
        nps = feats.get('notes_per_second', 0)
        ratio = effa / max(dens, 0.1)
        redun = effa / max(rcnps, 0.1)
        print(f'  nps={nps:.2f} rcnps={rcnps:.2f} dens={dens:.2f} effa={effa:.2f} effp={effp:.0f}')
        print(f'  eff_density_ratio={ratio:.3f}  eff_avg/rcnps={redun:.3f}')
        print(f'  mf3={feats.get("multi_finger_3plus_events",0)} mf4={feats.get("multi_finger_4plus_events",0)} wmf={feats.get("weighted_mf_score_per_sec",0):.2f}')
        # 方案B重算
        r2 = eff_density_features(cd)
        if r2:
            print(f'  [方案B] above_avg原始={r2["above_avg_orig"]:.2f} -> eff版={r2["above_avg_eff"]:.2f} (降幅={(1-r2["above_avg_eff"]/max(r2["above_avg_orig"],0.5))*100:.0f}%)')
    except Exception as e:
        print('  错误:', e)
    print()
