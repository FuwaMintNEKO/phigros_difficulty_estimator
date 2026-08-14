import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import json

path = os.path.join(_ROOT, 'data', 'chart', '4641132726938698.json')
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

lines = data.get('judgeLineList', [])

# Check line-level keys for notes
for i, line in enumerate(lines):
    if line.get('numOfNotes', 0) > 0:
        print(f'线 {i}:')
        print(f'  Line 层所有 keys: {list(line.keys())}')
        
        # Check if there's a 'notes' key directly on the line
        for extra_key in ['notes', 'note', 'data']:
            if extra_key in line:
                print(f'  发现 {extra_key}! {len(line[extra_key])}条')
        
        # Check eventLayers more carefully - maybe notes are in numOfNotes
        el = line.get('eventLayers', [])
        if len(el) > 0:
            for k, v in el[0].items():
                if isinstance(v, list):
                    # Check if any item has a 'type' field
                    has_type = any(isinstance(x, dict) and 'type' in x for x in v[:10])
                    print(f'    {k}: {len(v)}条, has_type={has_type}')
                    if v and isinstance(v[0], dict):
                        print(f'      键: {list(v[0].keys())}')
                        if len(v) > 1:
                            print(f'      第一条: {list(v[0].values())[:5]}')
                            print(f'      第二条: {list(v[1].values())[:5]}')
        
        # Check if there are extra layers
        print(f'\n  eventLayers数量: {len(el)}')
        
        # Let me check all items across all keys to find "note-like" data
        for layer_idx, layer in enumerate(el):
            for k, v in layer.items():
                if isinstance(v, list):
                    for item in v[:3]:
                        if isinstance(item, dict):
                            if 'type' in item or 'noteType' in item or 'position' in item:
                                print(f'  Layer{layer_idx}.{k}: NOTE FOUND! type={item.get("type","?")}')
        
        break
