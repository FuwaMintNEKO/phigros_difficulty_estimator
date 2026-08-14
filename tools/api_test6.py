# -*- coding: utf-8 -*-
"""验证新Flask版本 + 用户场景"""
import os, sys, io, json, urllib.request, uuid
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
    boundary = '----X' + uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="level"\r\n\r\nIN\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="speed"\r\n\r\n1.0\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{nm}.json"\r\nContent-Type: application/json\r\n\r\n'
    ).encode() + data + f'\r\n--{boundary}--\r\n'.encode()
    req = urllib.request.Request('http://127.0.0.1:5000/predict', data=body, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read().decode('utf-8'))
    r = result['results'][0]
    print(f'{nm}: prediction={r["prediction"]} gb={r["gb"]} boost={r["boost"]} version={r["version"]}')
print('DONE')