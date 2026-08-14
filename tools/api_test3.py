# -*- coding: utf-8 -*-
"""通过 /predict (multipart表单) 复现用户场景: IN档"""
import os, sys, io, json, urllib.request, uuid
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
cases = [
    ('夢降日(双指)', os.path.join(DL, '夢の降る日に', '5333883479687925.json')),
    ('DerSchneid(多指)', os.path.join(DL, 'Der Schneid(1)', '1903581575578621.json')),
]
def multipart(fname, data):
    boundary = '----WebKitFormBoundary' + uuid.uuid4().hex
    parts = []
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="level"\r\n\r\nIN'.encode())
    parts.append(f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="speed"\r\n\r\n1.0'.encode())
    parts.append(f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{fname}"\r\nContent-Type: application/json\r\n\r\n'.encode())
    parts.append(data)
    parts.append(f'\r\n--{boundary}--\r\n'.encode())
    body = b''.join(parts)
    return body, f'multipart/form-data; boundary={boundary}'

for nm, p in cases:
    with open(p, 'rb') as f:
        data = f.read()
    body, ctype = multipart(os.path.basename(p), data)
    req = urllib.request.Request('http://127.0.0.1:5000/predict', data=body, headers={'Content-Type': ctype})
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        result = json.loads(resp.read().decode('utf-8'))
        r0 = result[0] if isinstance(result, list) else result
        print(f'{nm}: prediction={r0.get("prediction")} gb={r0.get("gb")} boost={r0.get("boost")} level_used={r0.get("level_used")} version={r0.get("version")}')
    except Exception as e:
        print(f'{nm} API失败: {e}')
print('DONE')