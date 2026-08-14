import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os; sys.path.insert(0, '.')
import json

CHART_DIR = os.path.join(_ROOT, 'data', 'chart')

# 看标准谱的 positionX 范围
fp = os.path.join(CHART_DIR, 'Cthugha.USAO.0', 'AT.json')
with open(fp) as f:
    data = json.load(f)

all_x = []
for jl in data['judgeLineList']:
    for n in jl.get('notesAbove', []):
        all_x.append(n['positionX'])

print(f'Cthugha AT positionX: min={min(all_x):.2f}, max={max(all_x):.2f}, mean={sum(all_x)/len(all_x):.2f}')

# 看 PE 解析后的值
sys.path.insert(0, '.')
from unified_parser import _parse_pe_format

pe_fp = os.path.join(CHART_DIR, 'test_datas', '80116145.json')
with open(pe_fp) as f:
    text = f.read()
pe_data = _parse_pe_format(text)

all_x_pe = []
for jl in pe_data['judgeLineList']:
    for n in jl.get('notesAbove', []):
        all_x_pe.append(n['positionX'])

print(f'CrazyTek PE positionX: min={min(all_x_pe):.2f}, max={max(all_x_pe):.2f}, mean={sum(all_x_pe)/len(all_x_pe):.2f}')
