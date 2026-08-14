# -*- coding: utf-8 -*-
"""直接读原始JSON: 全部 tap(type=1) 的 startTime 拍间隔 (官方语义 beat=m+b/d)"""
import os, sys, io, json, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
p = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star', '61184.json')
raw = json.load(open(p, encoding='utf-8'))

def beat_of(st):
    return st[0] + st[1]/max(st[2], 1)

taps = []
all_types = {}
for jl in raw.get('judgeLineList', []):
    for n in jl.get('notes', []):
        ty = n.get('type')
        all_types[ty] = all_types.get(ty, 0) + 1
        st = n.get('startTime')
        if isinstance(st, list) and len(st) >= 3 and ty == 1:
            taps.append(beat_of(st))
        if isinstance(st, list) and len(st) >= 3 and ty == 2:
            pass  # hold
print('类型分布:', all_types)
print('tap 总数:', len(taps))

# 去掉重复(同拍多押)后排序
taps_sorted = sorted(set(taps))
print(f'tap 去重后: {len(taps_sorted)}')

# 间隔(拍)
d = np.diff(np.array(taps_sorted))
print(f'\n拍间隔: min={d.min():.4f} P10={np.percentile(d,10):.4f} P25={np.percentile(d,25):.4f} P50={np.percentile(d,50):.4f} P75={np.percentile(d,75):.4f}')
# 分音定义: 24分=1/6拍=0.1667, 16分=1/4=0.25, 8分=1/2=0.5, 4分=1.0
n24 = np.sum(d <= 0.167 + 1e-6)
n16 = np.sum((d > 0.167) & (d <= 0.25 + 1e-6))
n8 = np.sum((d > 0.25) & (d <= 0.5 + 1e-6))
n4 = np.sum((d > 0.5) & (d <= 1.0 + 1e-6))
n_rest = np.sum(d > 1.0)
tot = len(d)
print(f'\n=== 原始JSON tap 间隔(去重后) ===')
print(f'总间隔: {tot}')
print(f'24分(<=0.167拍): {n24} ({n24/tot*100:.1f}%)')
print(f'16分(0.167-0.25): {n16} ({n16/tot*100:.1f}%)')
print(f'8分(0.25-0.5): {n8} ({n8/tot*100:.1f}%)')
print(f'4分(0.5-1.0): {n4} ({n4/tot*100:.1f}%)')
print(f'更宽(>1.0): {n_rest} ({n_rest/tot*100:.1f}%)')
print(f'\n16分+24分: {n16+n24} ({ (n16+n24)/tot*100:.1f}%)')
print(f'16分+24分+8分: {n16+n24+n8} ({ (n16+n24+n8)/tot*100:.1f}%)')

# 原始间隔序列前60
print(f'\n原始拍间隔序列(前60):')
print(' '.join(f'{x:.3f}' for x in d[:60]))
print('DONE')