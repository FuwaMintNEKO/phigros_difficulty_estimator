# -*- coding: utf-8 -*-
"""解压两个zip并预测: Der Schneid / 夢の降る日に + 官谱原谱对比"""
import os, sys, io, zipfile, glob, shutil, tempfile
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
import app as app_mod
from unified_parser import load_chart_from_bytes
from feature_extractor import extract_features

DL = r'C:\Users\NaNK\Downloads'
tmp = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
os.makedirs(tmp, exist_ok=True)

# 解压
for z in ['Der Schneid(1).zip', '夢の降る日に.zip']:
    zp = os.path.join(DL, z)
    dst = os.path.join(tmp, z.replace('.zip', ''))
    if os.path.exists(zp):
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(dst)
        print(f'解压 {z} ->', dst)
        for f in os.listdir(dst):
            print('   ', f)
# 找 json 文件
targets = []
for root, dirs, files in os.walk(tmp):
    for f in files:
        if f.endswith('.json'):
            targets.append(os.path.join(root, f))
print('\njson文件:', targets)

def predict_file(path, level='AT'):
    with open(path, 'rb') as f:
        cd, raw = load_chart_from_bytes(f.read())
    if cd is None:
        return None, '解析失败'
    r, err = app_mod.predict_one_chart(cd, speed=1.0, level=level, is_custom=True, chart_name=os.path.basename(path))
    if r is None:
        return None, err
    return r, None

for t in targets:
    r, err = predict_file(t)
    if r:
        print(f'\n=== {os.path.basename(t)} ===')
        print(f'  预测={r["prediction"]} gb={r["gb"]} boost={r["boost"]}')
        print(f'  mf3={r["categories"]}')
        print(f'  tags={r["tags"]}')
        print(f'  nps={r["real_notes_per_second"]} core_nps={r["real_core_notes_per_second"]} notes={r["total_notes"]}')
    else:
        print(f'  {t} 失败: {err}')
print('DONE')