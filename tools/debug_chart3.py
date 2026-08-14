import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import json

path = os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json')
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

lines = data.get('judgeLineList', [])

# Find a line with notes and dump the full eventLayers structure
for i, line in enumerate(lines):
    if line.get('numOfNotes', 0) > 0:
        print(f'--- 线 {i}: numOfNotes={line["numOfNotes"]} ---')
        el = line.get('eventLayers', [])
        print(f'  eventLayers count: {len(el)}')
        for j, layer in enumerate(el):
            print(f'\n  Layer {j} keys: {list(layer.keys())}')
            for k, v in layer.items():
                if isinstance(v, list) and len(v) > 0:
                    item = v[0]
                    if isinstance(item, dict):
                        print(f'    {k}: {len(v)}条, keys={list(item.keys())}')
                    else:
                        print(f'    {k}: {len(v)}条, type={type(item).__name__}, 第一条={item}')
        break

# Check BPMList format
print(f'\nBPMList: {data.get("BPMList")}')
print(f'META: {data.get("META")}')
print(f'chartTime: {data.get("chartTime")}')
