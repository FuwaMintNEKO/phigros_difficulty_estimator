# -*- coding: utf-8 -*-
"""验证时间转换bug: predict_rpe.py vs RPE官方语义"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
raw = json.load(open(p, encoding='utf-8'))

# RPE官方 (rephiedit.ts): startBeat = m + b/d  (m就是拍!)
def official_beat(st):
    return st[0] + st[1]/max(st[2],1)

# predict_rpe.py: (m*4 + b*(4/d))*8
def predict_rpe_tick(st):
    return (st[0]*4.0 + st[1]*(4.0/st[2])) * 8.0

# _normalize_time_st: (m + b/d)*32
def normalize_tick(st):
    return (st[0] + st[1]/st[2]) * 32

# 验证: tap [111,3,4] = 111.75拍
st = [111, 3, 4]
print(f'startTime={st}:')
print(f'  官方beat = {official_beat(st)} 拍')
print(f'  predict_rpe tick = {predict_rpe_tick(st)} (= {predict_rpe_tick(st)/32}拍 @1拍32tick)')
print(f'  normalize tick = {normalize_tick(st)} (= {normalize_tick(st)/32}拍)')
print()
# BPM事件: [256,0,1] → 官方256拍 vs predict_rpe 8192tick=256拍? 8192/32=256 ✓
# 但 predict_rpe 的公式是 (256*4)*8=8192 → 如果1拍=8tick, 8192=1024拍!
print('=== 关键矛盾 ===')
print('predict_rpe: [256,0,1] → (256*4)*8 = 8192')
print('  如果按1拍=32tick(官谱): 8192/32 = 256拍 ✓ (碰巧对!)')
print('  因为 ×4(小节→拍) 和 ×8(拍→tick) 组合 = ×32')
print('  m*4*8 = m*32, 而官方应该是 m*32 (m就是拍, 1拍=32tick)')
print('  → b*(4/d)*8 = b*32/d, 官方应该是 b*32/d ✓')
print('  所以 predict_rpe 公式数值上碰巧正确!')
print()
# 那问题在哪? 再验证 _normalize_time_st
print('=== _normalize_time_st ===')
print('  beat = m + b/d, tick = beat*32 → [111,3,4] = 111.75*32 = 3576 tick')
print('  predict_rpe → [111,3,4] = (111*4 + 3)*8 = 3576 tick')
print('  两者一致!')
print()
# 但 collect_all_notes 用哪个? 看转换后音符是否有 time 字段
from unified_parser import load_chart_from_bytes
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
# cd 是 convert_rpe_to_standard 的结果 (音符有 time)
jls = cd.get('judgeLineList', [])
# 找高潮段 tap 的 time
tap_times = []
for jl in jls:
    for n in jl.get('notesAbove', []):
        if n.get('type') == 1:
            tap_times.append(n['time'])
tap_times.sort()
if len(tap_times) > 1:
    d = np.diff(np.array(tap_times))
    print(f'\n转换后 tap time 间隔(tick): min={d.min():.1f} P25={np.percentile(d,25):.1f} P50={np.percentile(d,50):.1f}')
    print(f'  16分应=8 tick, 8分应=16 tick, 24分应=5.33 tick')
    print(f'  <=8 tick: {np.sum(d<=8)}')
    print(f'  序列前20: {np.round(d[:20],1)}')
print('DONE')