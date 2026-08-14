# -*- coding: utf-8 -*-
"""锚点谱: Apollo/Chart_SP 用 predict_one_chart (返回tuple)"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
from unified_parser import load_chart_from_bytes

for cid, path, lv, label in [
    (41242, os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '41242.json'), 'AT', 'Apollo(未上架, 锚点18.0)'),
    (None, os.path.join(_ROOT, 'data', 'test_charts', 'Chart_SP #1347(1).json'), 'IN', 'Chart_SP#1347(锚点17.6+)'),
    (None, os.path.join(_ROOT, 'data', 'test_charts', 'Apollo(17.8).json'), 'AT', 'Apollo(test_charts文件, 文件名标17.8)'),
]:
    with open(path, 'rb') as f:
        cd, raw = load_chart_from_bytes(f.read())
    r, err = app_mod.predict_one_chart(cd, speed=1.0, level=lv, is_custom=True, chart_name=os.path.basename(path))
    if r is None:
        print(label, '失败:', err)
    else:
        print(f"{label}: pred={r['prediction']} gb={r['gb']} boost={r['boost']} hold%={r.get('total_notes') and ''}")
        print(f"   tags={r['tags']} notes={r['total_notes']} dur={r['duration_sec']}s nps={r['real_notes_per_second']} core_nps={r['real_core_notes_per_second']} mf3={r['categories']}")
print('DONE')