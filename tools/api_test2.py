# -*- coding: utf-8 -*-
"""通过 /predict_one 复现用户场景: ?level=IN"""
import os, sys, io, json, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
cases = [
    ('夢降日(双指)', os.path.join(DL, '夢の降る日に', '5333883479687925.json')),
    ('DerSchneid(多指)', os.path.join(DL, 'Der Schneid(1)', '1903581575578621.json')),
]
for nm, p in cases:
    with open(p, 'rb') as f:
        data = f.read()
    req = urllib.request.Request('http://127.0.0.1:5000/predict_one?level=IN', data=data, headers={'Content-Type': 'application/octet-stream'})
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        result = json.loads(resp.read().decode('utf-8'))
        print(f'{nm}: prediction={result.get("prediction")} gb={result.get("gb")} boost={result.get("boost")} level_used={result.get("level_used")} version={result.get("version")}')
    except Exception as e:
        print(f'{nm} API失败: {e}')
print('DONE')