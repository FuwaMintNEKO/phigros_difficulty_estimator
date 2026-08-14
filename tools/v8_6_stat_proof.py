"""
v8.6 STATISTICAL PROOF: 数学/统计学证明脚本
===========================================
证明/证伪以下假设:
  H1: rcnps 与 density_dimension 高度冗余 (r>0.95)
  H2: speed_volatility 与定数零相关 (|r|<0.05)
  H3: 配置维度 co 总和 (0.36) 显著低于耐力 (0.44)，需要再平衡
  H4: above_avg_duration_sec 与定数显著正相关，有独立信息
  H5: trill_density 与定数正相关，可为配置维度补充信息
  H6: 几何平均 vs 算术平均: 哪个 density_dimension 公式更优
  H7: rest_ratio/drag_flick_ratio/pattern_switch_rate 是否真的负相关
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import sys, os, pickle, numpy as np, math, re
sys.path.insert(0, '.')
from feature_extractor import extract_features
from data_loader import load_difficulty_tsv, find_chart_files, load_chart_json
from unified_parser import load_chart_from_bytes
from sklearn.linear_model import LinearRegression
import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print('='*70)
print('  v8.6 STATISTICAL PROOF — 数学证明')
print('='*70)

# ===== 加载数据 =====
CHART_DIR = os.path.join(_ROOT, 'data', 'chart')
TSV = os.path.join(_ROOT, 'data', 'info', 'difficulty.tsv')
diff_map = load_difficulty_tsv(TSV)
chart_files = find_chart_files(CHART_DIR)

all_official = []
for fn, info in chart_files.items():
    sid = info['song_id']
    if sid not in diff_map: continue
    for lv in ['IN','AT']:
        if lv not in info.get('levels',{}): continue
        if lv not in diff_map[sid]: continue
        try:
            cd = load_chart_json(info['levels'][lv]); f = extract_features(cd)
            if f:
                f['_difficulty'] = diff_map[sid][lv]; f['_name'] = fn[:30]
                all_official.append(f)
        except: pass

exclude = ['chartnekockLK','snow dance','Snow']
all_official = [f for f in all_official if not any(p.lower() in f['_name'].lower() for p in exclude)]
print(f'官谱: {len(all_official)}')

diffs = np.array([f['_difficulty'] for f in all_official])
names = [f['_name'] for f in all_official]

def r_with_diff(feature_name):
    vals = np.array([f.get(feature_name, 0) for f in all_official], dtype=float)
    if np.std(vals) < 1e-8: return 0.0, vals
    r = np.corrcoef(vals, diffs)[0,1]
    return r, vals

# ============================================================
# H1: rcnps vs density_dimension 冗余性
# ============================================================
print('\n' + '='*70)
print('H1: rcnps 与 density_dimension 高度冗余')
print('='*70)

r_rcnps, vals_rcnps = r_with_diff('real_core_notes_per_second')
r_dd, vals_dd = r_with_diff('density_dimension')
r_cross = np.corrcoef(vals_rcnps, vals_dd)[0,1]

print(f'  rcnps <-> 定数:              r = {r_rcnps:.4f}')
print(f'  density_dimension <-> 定数:  r = {r_dd:.4f}')
print(f'  rcnps <-> density_dimension:  r = {r_cross:.4f}')

# 偏相关: 控制 density_dimension 后 rcnps 的独立贡献
from scipy import stats
# 简单方法: 线性回归残差
residual = vals_rcnps - np.polyval(np.polyfit(vals_dd, vals_rcnps, 1), vals_dd)
r_residual = np.corrcoef(residual, diffs)[0,1]
print(f'  rcnps 在控制 density_dimension 后的偏相关: r = {r_residual:.4f}')

if r_cross > 0.95:
    print(f'  >> 结论: r={r_cross:.4f} > 0.95, 高度冗余. 建议移除 rcnps, 合并 co 到 density_dimension')
else:
    print(f'  >> 结论: r={r_cross:.4f} < 0.95, 保留独立信息')

# ============================================================
# H2: speed_volatility 零相关
# ============================================================
print('\n' + '='*70)
print('H2: speed_volatility 与定数零相关')
print('='*70)

r_sv, vals_sv = r_with_diff('speed_volatility')
print(f'  speed_volatility <-> 定数:  r = {r_sv:.4f}')
print(f'  |r| = {abs(r_sv):.4f}')

if abs(r_sv) < 0.05:
    print(f'  >> 结论: |r|={abs(r_sv):.4f} < 0.05, 与噪声无异. 建议从 FLAT 移除')
else:
    print(f'  >> 结论: |r|={abs(r_sv):.4f} >= 0.05, 可能有微弱信号, 保留观察')

# ============================================================
# H3: 配置维度权重失衡
# ============================================================
print('\n' + '='*70)
print('H3: 配置维度权重失衡分析')
print('='*70)

# 当前 v8.5 FLAT 配置和耐力特征的 co 和 r 值
config_current = [
    ('stair_rate_per_sec', 0.05, 2.0),
    ('stair_complexity', 0.02, 0.2),
    ('chord_size_entropy', 0.02, 0.5),
    ('chord_alternation_rate', 0.08, 0.5),
    ('weighted_mf_score_per_sec', 0.05, 10.0),
    ('position_entropy', 0.02, 2.0),
    ('avg_chord_size_poly', 0.03, 2.0),
    ('drag_flick_ratio', 0.02, 0.2),
    ('pattern_switch_rate', 0.05, 1.0),
    ('position_range_used', 0.02, 0.5),
]
stamina_current = [
    ('above_avg_density_mean', 0.25, 4.0),
    ('total_notes', 0.15, 400.0),
    ('tap_burst_top5', 0.04, 0.5),
]

config_co_sum = sum(c for _,c,_ in config_current)
stamina_co_sum = sum(c for _,c,_ in stamina_current)
print(f'  配置维度 co 总和: {config_co_sum:.2f} (10个特征)')
print(f'  耐力维度 co 总和: {stamina_co_sum:.2f} (3个特征)')
print(f'  配置平均 co/特征:  {config_co_sum/len(config_current):.4f}')
print(f'  耐力平均 co/特征:  {stamina_co_sum/len(stamina_current):.4f}')

print(f'\n  各配置特征 r 值:')
for name, co, bl in config_current:
    r, _ = r_with_diff(name)
    print(f'    {name:<35s} co={co:.2f}  r={r:+.4f}')

print(f'\n  各耐力特征 r 值:')
for name, co, bl in stamina_current:
    r, _ = r_with_diff(name)
    print(f'    {name:<35s} co={co:.2f}  r={r:+.4f}')

print(f'  >> 结论: 配置 co 总和 {config_co_sum:.2f} < 耐力 {stamina_co_sum:.2f}')
print(f'     但配置特征数 10 > 耐力 3, 配置平均 co 仅 {config_co_sum/len(config_current):.4f}')

# ============================================================
# H4: above_avg_duration_sec 独立信息
# ============================================================
print('\n' + '='*70)
print('H4: above_avg_duration_sec 独立信息量')
print('='*70)

r_ad, vals_ad = r_with_diff('above_avg_duration_sec')
r_aam, vals_aam = r_with_diff('above_avg_density_mean')
r_cross_ad_aam = np.corrcoef(vals_ad, vals_aam)[0,1]

print(f'  above_avg_duration_sec <-> 定数:     r = {r_ad:.4f}')
print(f'  above_avg_density_mean <-> 定数:    r = {r_aam:.4f}')
print(f'  duration_sec <-> density_mean:      r = {r_cross_ad_aam:.4f}')

# 偏相关: 控制 above_avg_density_mean 后 duration_sec 的独立贡献
residual_ad = vals_ad - np.polyval(np.polyfit(vals_aam, vals_ad, 1), vals_aam)
r_ad_residual = np.corrcoef(residual_ad, diffs)[0,1]
print(f'  duration_sec 在控制 density_mean 后的偏相关: r = {r_ad_residual:.4f}')

if r_ad > 0.15 and r_ad_residual > 0.05:
    print(f'  >> 结论: 有显著独立信息(r={r_ad:.4f}, 偏r={r_ad_residual:.4f}), 建议加入 FLAT')
elif r_ad > 0.15:
    print(f'  >> 结论: r={r_ad:.4f} 但偏r={r_ad_residual:.4f}小, 独立信息有限, 可加但降co')

# ============================================================
# H5: trill_density 独立信息
# ============================================================
print('\n' + '='*70)
print('H5: trill_density 独立信息量')
print('='*70)

r_td, vals_td = r_with_diff('trill_density')
r_srs, vals_srs = r_with_diff('stair_rate_per_sec')
r_cross_td_srs = np.corrcoef(vals_td, vals_srs)[0,1]

print(f'  trill_density <-> 定数:          r = {r_td:.4f}')
print(f'  stair_rate_per_sec <-> 定数:     r = {r_srs:.4f}')
print(f'  trill <-> stair:                 r = {r_cross_td_srs:.4f}')

# 偏相关
residual_td = vals_td - np.polyval(np.polyfit(vals_srs, vals_td, 1), vals_srs)
r_td_residual = np.corrcoef(residual_td, diffs)[0,1]
print(f'  trill 在控制 stair 后的偏相关:  r = {r_td_residual:.4f}')

if r_td > 0.1:
    print(f'  >> 结论: r={r_td:.4f}, 有独立信息, 建议加入 FLAT (配置维度)')

# ============================================================
# H6: 几何平均 vs 算术平均 density_dimension
# ============================================================
print('\n' + '='*70)
print('H6: density_dimension 公式对比')
print('='*70)

# 当前: geometric = sqrt(rcnps * above_avg_density_mean)
# 候选: arithmetic = (rcnps + above_avg_density_mean) / 2
# 候选: harmonic = 2 * rcnps * above_avg_density_mean / (rcnps + above_avg_density_mean)

geo_vals = np.sqrt(vals_rcnps * vals_aam)
ari_vals = (vals_rcnps + vals_aam) / 2
# 避免除零
mask = (vals_rcnps + vals_aam) > 0
har_vals = np.zeros_like(geo_vals)
har_vals[mask] = 2 * vals_rcnps[mask] * vals_aam[mask] / (vals_rcnps[mask] + vals_aam[mask])

r_geo = np.corrcoef(geo_vals, diffs)[0,1]
r_ari = np.corrcoef(ari_vals, diffs)[0,1]
r_har = np.corrcoef(har_vals, diffs)[0,1]

print(f'  几何平均 (sqrt)  <-> 定数:  r = {r_geo:.4f}  (当前)')
print(f'  算术平均 (mean)  <-> 定数:  r = {r_ari:.4f}')
print(f'  调和平均 (harm)  <-> 定数:  r = {r_har:.4f}')

best = max(r_geo, r_ari, r_har)
if r_ari == best:
    print(f'  >> 结论: 算术平均 r={r_ari:.4f} 最优, 建议切换')
elif r_geo == best:
    print(f'  >> 结论: 几何平均 r={r_geo:.4f} 最优, 保持当前')
else:
    print(f'  >> 结论: 调和平均 r={r_har:.4f} 最优, 但需要验证')

# ============================================================
# H7: 负相关特征验证
# ============================================================
print('\n' + '='*70)
print('H7: 负相关特征验证')
print('='*70)

for name in ['rest_ratio', 'drag_flick_ratio', 'pattern_switch_rate', 
             'above_below_cross', 'above_avg_density_ratio', 'stamina_ratio']:
    r, _ = r_with_diff(name)
    status = '✗ 负相关!' if r < -0.05 else ('✓ 正相关' if r > 0.05 else '~ 零相关')
    print(f'  {name:<35s} r = {r:+.4f}  {status}')

# ============================================================
# 额外: jack_density 分析
# ============================================================
print('\n' + '='*70)
print('额外: jack_density 分析')
print('='*70)

r_jd, vals_jd = r_with_diff('jack_density')
r_tap, vals_tap = r_with_diff('tap_per_second')
r_cross_jd_tap = np.corrcoef(vals_jd, vals_tap)[0,1]

print(f'  jack_density <-> 定数:        r = {r_jd:.4f}')
print(f'  jack_density <-> tap_per_sec: r = {r_cross_jd_tap:.4f}')

# 偏相关
residual_jd = vals_jd - np.polyval(np.polyfit(vals_tap, vals_jd, 1), vals_tap)
r_jd_residual = np.corrcoef(residual_jd, diffs)[0,1]
print(f'  jack 在控制 tap_per_sec 后的偏相关: r = {r_jd_residual:.4f}')

# ============================================================
# 额外: 多指特征分析
# ============================================================
print('\n' + '='*70)
print('额外: 已存在但未在FLAT的特征 r 值')
print('='*70)

extra_features = [
    'above_avg_duration_sec', 'trill_density', 'jack_density',
    'stair_chord_ratio', 'multi_finger_3plus_events', 'discrete_mf_ratio',
    'tap_per_second', 'duration_sec', 'offbeat_ratio',
    'stair_density', 'stair_speed_avg',
    'simultaneous_event_count', 'simultaneous_ratio',
    'global_jack_count', 'burst_intensity_mean',
    'real_active_sec', 'real_notes_per_second',
    'direction_irregularity', 'track_deviation_score',
]
for name in extra_features:
    r, _ = r_with_diff(name)
    status = '✓' if r > 0.15 else ('~' if r > 0.05 else '✗')
    print(f'  {status} {name:<35s} r = {r:+.4f}')

print('\n' + '='*70)
print('  证明完成')
print('='*70)