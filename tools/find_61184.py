# -*- coding: utf-8 -*-
"""找 #61184 元数据 + 预测"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
meta = json.load(open(os.path.join(_ROOT, 'data', 'phira', 'unranked_all.json'), encoding='utf-8'))
for c in meta:
    if c['id'] == 61184:
        print('找到 #61184:')
        print(json.dumps(c, ensure_ascii=False, indent=1)[:1200])
        break
else:
    print('#61184 不在 unranked_all')
# 检查json文件是否存在
import os.path
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
print('\n文件存在:', os.path.exists(p))
print('DONE')