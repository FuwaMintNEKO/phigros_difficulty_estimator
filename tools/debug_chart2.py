import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import json

path = os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json')
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print('BPMList:', data.get('BPMList', 'N/A'))
print('META:', data.get('META', 'N/A'))
print('chartTime:', data.get('chartTime', 'N/A'))

# Look at first few eventLayers
lines = data.get('judgeLineList', [])
for i in range(min(5, len(lines))):
    line = lines[i]
    print(f'\n--- 第{i}条线 ---')
    print(f'  Group: {line.get("Group")}')
    print(f'  Name: {line.get("Name")}')
    print(f'  numOfNotes: {line.get("numOfNotes")}')
    print(f'  bpmfactor: {line.get("bpmfactor")}')
    print(f'  isCover: {line.get("isCover")}')
    
    el = line.get('eventLayers', [])
    print(f'  eventLayers数: {len(el)}')
    for j, layer in enumerate(el[:3]):
        print(f'    Layer {j} keys: {list(layer.keys())}')
        for k, v in layer.items():
            if isinstance(v, list) and len(v) > 0:
                print(f'      {k}: {len(v)}条, 第一条: {v[0]}')

# Check total notes via numOfNotes
total_num = sum(line.get('numOfNotes', 0) for line in lines)
print(f'\n所有线numOfNotes总和: {total_num}')

# Look at a line with numOfNotes > 0
for i, line in enumerate(lines):
    if line.get('numOfNotes', 0) > 0:
        print(f'\n--- 非空线 {i} ---')
        print(f'  numOfNotes: {line["numOfNotes"]}')
        el = line.get('eventLayers', [])
        print(f'  eventLayers数: {len(el)}')
        for j, layer in enumerate(el):
            print(f'  Layer {j} keys: {list(layer.keys())}')
            for k, v in layer.items():
                if isinstance(v, list):
                    print(f'    {k}: {len(v)}条')
                    if v:
                        print(f'    第一条: {v[0]}')
                        if len(v) > 1:
                            print(f'    第二条: {v[1]}')
        break
