# -*- coding: utf-8 -*-
"""AT段domain align验证: 计算AT段delta, 模拟对齐效果"""
import os, sys, pickle, numpy as np, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from boost_config import MANUAL_FLAT

# 官谱 AT段 密度类特征
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)

# 官谱 >=16.5
off_at = [f for f in cache['official'] if f['diff'] >= 16.5]
# 上架 >=16.5
rkd_at = [r for r in cache['ranked'] if r['diff'] and r['diff'] >= 16.5]
print(f'官谱>=16.5: {len(off_at)} | 上架>=16.5: {len(rkd_at)}')

# 现有 domain_align 的 IN delta 特征列表
align = json.load(open(os.path.join(_ROOT, 'data', 'domain_align.json'), encoding='utf-8'))
D = list(align['delta'].keys())
print(f'现有delta特征: {len(D)}')

# AT段 delta (密度类特征)
delta_at = {}
for k in D:
    a = [f['feats'].get(k, 0) for f in off_at]
    b = [r['feats'].get(k, 0) for r in rkd_at]
    if np.mean(a) > 1e-6 or np.mean(b) > 1e-6:
        d = np.mean(b) - np.mean(a)
        if abs(d) > 0.02 * max(abs(np.mean(a)), 1e-6):
            delta_at[k] = float(d)
print(f'AT段显著delta特征: {len(delta_at)}')
# 显示最大的几个
top = sorted(delta_at.items(), key=lambda x: -abs(x[1]))[:15]
for k, v in top:
    a = np.mean([f['feats'].get(k, 0) for f in off_at])
    b = np.mean([r['feats'].get(k, 0) for r in rkd_at])
    print(f'  {k:<32} 官={a:>9.2f} 上={b:>9.2f} delta={v:>+8.2f}')

# 模拟: AT段上架谱 减 delta_at 后重新预测 (用v11.1模型+条件boost+校准)
with open(os.path.join(_ROOT, 'models', '6dim_model_v11_1.pkl'), 'rb') as f:
    m = pickle.load(f)
gb, scaler = m['gb'], m['scaler']
FN, P95, P99 = m['feature_names'], m['p95_vals'], m['p99_vals']
LV_ORDER = m.get('lv_order', ['EZ','HD','IN','AT'])
CAPS = m.get('caps', {})
FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT)
MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}

def predict(feats_raw, level='IN', align_delta=None):
    feats = dict(feats_raw)
    if align_delta:
        for k, d in align_delta.items():
            if k in feats: feats[k] = feats[k] - d
    lv = level.upper()
    if 'IN_AT' in LV_ORDER and lv in ('IN','AT'): lv = 'IN_AT'
    if lv not in LV_ORDER: lv = LV_ORDER[-1]
    vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(lv)] = 1.0
    x = np.array([[feats.get(n,0) for n in FN] + vec])
    xs = scaler.transform(x)
    p_gb = float(gb.predict(xs)[0])
    mf3 = feats_raw.get('multi_finger_3plus_events', 0)
    dens = feats_raw.get('above_avg_density_mean', 0)
    mf_scale = (0.70 if dens >= 12.5 else 0.50) if mf3 >= 30 else (1.0 if mf3 <= 5 else 0.8)
    eff_scale = 1.0 if mf3 >= 30 else (1.5 if mf3 <= 5 else 1.0)
    total = 0.0
    cd_ = CAPS.get('_default', None)
    for fname, bl, co in FLAT:
        v = feats.get(fname, 0)
        pv = P95.get(fname, 0)
        t = max(pv*0.55, bl*0.5)
        if v <= t: continue
        e = v/t - 1.0
        c = CAPS.get(fname, cd_)
        if c is not None and e > c: e = c
        co2 = co * (mf_scale if fname in MF_FEATS else (eff_scale if fname in EFF_FEATS else 1.0))
        x_ = co2 * (e**0.70)
        p99 = max(P99.get(fname,0), bl*0.5)
        if v > p99:
            pe = v/p99 - 1.0
            if c is not None and pe > c: pe = c
            x_ += co2*max(0,pe)**0.70*0.5
        total += x_
    pred = p_gb + total
    for lo, hi, adj in [(14,15,0.30),(15,16,0.18),(16,17,0.05)]:
        if lo < pred <= hi: pred -= adj; break
    return pred

# 对比: AT段 无对齐 vs IN对齐(现状) vs AT对齐
print('\n=== AT段(>=16.5) 上架谱偏差对比 ===')
for vname, ad in [('无对齐', None), ('现状(无AT对齐)', None), ('AT段对齐', delta_at)]:
    if vname == '现状(无AT对齐)':
        # 现状: IN段对齐(level IN时) + 校准
        errs = [predict(r['feats'], r['level'], None) - r['diff'] for r in rkd_at]
    else:
        errs = [predict(r['feats'], r['level'], ad) - r['diff'] for r in rkd_at]
    errs = np.array(errs)
    print(f'  {vname:<16} bias={errs.mean():+.3f} MAE={np.abs(errs).mean():.3f}')
