# -*- coding: utf-8 -*-
"""打印完整API响应"""
import os, sys, io, json, urllib.request, uuid
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DL = os.path.join(_ROOT, 'tools', '_tmp_dl_charts')
p = os.path.join(DL, '夢の降る日に', '5333883479687925.json')
with open(p, 'rb') as f:
    data = f.read()
boundary = '----X' + uuid.uuid4().hex
body = (
    f'--{boundary}\r\nContent-Disposition: form-data; name="level"\r\n\r\nIN\r\n'
    f'--{boundary}\r\nContent-Disposition: form-data; name="speed"\r\n\r\n1.0\r\n'
    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="test.json"\r\nContent-Type: application/json\r\n\r\n'
).encode() + data + f'\r\n--{boundary}--\r\n'.encode()
req = urllib.request.Request('http://127.0.0.1:5000/predict', data=body, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
try:
    resp = urllib.request.urlopen(req, timeout=120)
    raw = resp.read().decode('utf-8')
    print('响应:', raw[:2000])
except Exception as e:
    print('失败:', e)
    try:
        print('body:', e.read().decode('utf-8')[:1000])
    except: pass
print('DONE')