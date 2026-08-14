# -*- coding: utf-8 -*-
"""tick基准: 官谱1拍=32tick 验证"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes

# 官谱: 238bpm, 16分=63ms, 1拍=252ms=32tick → 16分=8tick? 不对
# 16分 = 1拍/4 = 8 tick (32/4)! 不是2!
print('官谱: 1拍=32 tick, 16分=32/4=8 tick, 8分=16 tick')
print('官谱 Verrückt 间隔 P50=8 tick = 16分音符 ✓ (238bpm: 8tick=252/4=63ms ✓)')
print()
# RPE Melodiniq: 之前计算 tick = (m*4 + b*(4/d))*8
# 如果RPE 1拍=8 tick(基于BPM事件: 256小节=1024拍=8192tick → 8tick/拍)
# 但官谱是 32 tick/拍! 单位不同!
# RPE转换: predict_rpe.py 里 start_time = (m*4 + b*(4/d))*8
# [256,0,1] → (256*4 + 0)*8 = 8192 ✓ (BPM事件显示8192)
# 官谱 1拍=32tick vs RPE 1拍=8tick → RPE转换×4?
print('\n=== RPE vs 官谱 tick 基准 ===')
print('RPE: startTime [m,b,d] → beat = m*4 + b*(4/d), tick = beat*8 → 1拍=8tick')
print('官谱: 1拍=32tick')
print('→ RPE 转换后 tick 应 ×4 才能对齐官谱!')
print('当前 predict_rpe: start_time = beat*8 (1拍=8tick) — 比官谱少4倍!')
# 验证: Melodiniq BPM事件 [256,0,1] → 转换后 8192tick, 官谱同位置应为 256*4拍*32tick=32768
# 检查 Melodiniq 转换后的 BPM 时间线
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
with open(p, 'rb') as f:
    cd, _ = load_chart_from_bytes(f.read())
bpm_tl = cd.get('BPMList', [])
print(f'\nMelodiniq 转换后 BPMList: {bpm_tl[:3]}')
# 官谱 BPM 事件 (如果是变速谱)
p2 = os.path.join(_ROOT, 'data', 'chart', 'Verruckt.Raimukun.0', 'IN.json')
raw2 = json.load(open(p2, encoding='utf-8'))
print(f'官谱 Verrückt BPMList: {raw2.get("BPMList", "无")}')
print('DONE')