# -*- coding: utf-8 -*-
"""#61184 Melodiniq 详细诊断"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, raw = load_chart_from_bytes(f.read())
r, err = app_mod.predict_one_chart(cd, speed=1.0, level='IN', is_custom=True, chart_name='Melodiniq')
if r:
    print(f'Melodiniq: prediction={r["prediction"]} gb={r["gb"]} boost={r["boost"]}')
    print(f'  tags={r["tags"]} nps={r["real_notes_per_second"]} core_nps={r["real_core_notes_per_second"]}')
    print(f'  key贡献:')
    for k in r['key_features'][:12]:
        print(f'    {k["name"]:<34} contrib={k["contribution"]:.3f} value={k["value"]} threshold={k["threshold"]} excess={k["excess"]}')
    # 特征
    feats = extract_features(cd, speed=1.0)
    print(f'\n特征: mf3={feats.get("multi_finger_3plus_events")} dens={feats.get("above_avg_density_mean"):.1f} '
          f'eff_peak={feats.get("eff_peak_tps_1s")} dur={feats.get("above_avg_duration_sec")} nps={feats.get("real_notes_per_second"):.1f}')
    print(f'  jline_mov={feats.get("jline_movement_density"):.1f} jline_rot={feats.get("jline_rotate_density"):.1f} '
          f'jline_dis={feats.get("jline_disappear_density"):.1f} ml={feats.get("multi_line_sim_events")}')
    print(f'  type_switch={feats.get("type_switch_per_sec"):.2f} wmf={feats.get("weighted_mf_score_per_sec"):.1f}')
    print(f'  drag={feats.get("drag_per_sec"):.2f} movement={feats.get("movement_per_second"):.1f}')
    # RPE 事件检查
    if isinstance(cd, dict):
        jls = cd.get('judgeLineList', [])
        tot = {'move': 0, 'rot': 0, 'dis': 0}
        for jl in jls:
            tot['move'] += len(jl.get('judgeLineMoveEvents', []))
            tot['rot'] += len(jl.get('judgeLineRotateEvents', []))
            tot['dis'] += len(jl.get('judgeLineDisappearEvents', []))
        print(f'\n解析后判定线事件: {tot}')
else:
    print('预测失败:', err)
print('DONE')