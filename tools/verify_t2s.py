# -*- coding: utf-8 -*-
"""验证 time_to_seconds 对间隔计算的 bug"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from feature_extractor import time_to_seconds
# Melodiniq bpm_timeline: 0-256拍=193, 256-320=196...
bpm_tl = [(0,193), (256,196), (320,206), (336,209), (352,212), (360,215), (368,218), (376,221), (384,224), (392,227), (400,230), (416,236), (480,240)]
# 16分间隔 8tick @ 240bpm段 (拍400-416)
# 正确: 8/32*60/240 = 62.5ms
print(f'正确 8tick@240 = {8/32*60/240*1000:.1f}ms')
# time_to_seconds 积分: target_beat=8/32=0.25拍 → 0~0.25 全在193段
t = time_to_seconds(8, 240.0, bpm_tl)
print(f'time_to_seconds(8tick,240) = {t*1000:.1f}ms (积分到0.25拍=193段)')
# 若间隔发生在拍400后, 绝对时间
t2 = time_to_seconds(400*32 + 8, 240.0, bpm_tl)
t3 = time_to_seconds(400*32, 240.0, bpm_tl)
print(f'绝对时间 t(400拍+8tick)={t2:.3f}s, t(400拍)={t3:.3f}s, 差值={t2-t3:.4f}s = {(t2-t3)*1000:.1f}ms')
print('\n=== 结论 ===')
print('diff间隔传入 time_to_seconds 积分模式: 返回从0开始的累计时间, 不是间隔!')
print('若diff很小(8tick), target_beat=0.25, 只积分0~0.25拍(193段) = 77.7ms (碰巧对)')
print('但若BPM段变化, diff跨段时积分也会错')
print('正确: 间隔秒 = tick/32 * 60/bpm (局部bpm)')
print('DONE')