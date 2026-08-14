# -*- coding: utf-8 -*-
"""v11.2 vs v11.8b 上架谱预测完整对比
"""
import os, sys, pickle, numpy as np, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
from scipy.stats import spearmanr
from boost_config import MANUAL_FLAT

def load_model(name):
    m = pickle.load(open(os.path.join(_ROOT, 'models', name), 'rb'))
    return m

def make_predictor(m, calib, version):
    gb, scaler = m['gb'], m['scaler']
    FN = m['feature_names']; LV_ORDER = m['lv_order']
    P95 = m['p95_vals']; P99 = m['p99_vals']
    FLAT = m.get('MANUAL_FLAT', MANUAL_FLAT); CAPS = m.get('caps', {})
    _ALIGN = json.load(open(os.path.join(_ROOT, 'data', 'domain_align.json'), encoding='utf-8')).get('delta', {})
    MF_FEATS = {'weighted_mf_score_per_sec', 'multi_finger_3plus_events', 'discrete_mf_ratio', 'chord_alternation_rate'}
    EFF_FEATS = {'eff_peak_tps_1s', 'eff_avg_tps_1s'}
    DENS_FEATS = {'above_avg_density_mean', 'real_core_notes_per_second'}
    EXTREME_FEATS = {'cross_hand_density', 'jline_relative_cross', 'thirtysecond_run_max', 'thirtysecond_run_ratio', 'lane_switch_density'}
    TAG_TH = json.load(open(os.path.join(_ROOT, 'data', 'tag_thresholds.json'), encoding='utf-8'))
    TAG_DIM = [('底力', 'above_avg_density_mean'), ('多押', 'weighted_mf_score_per_sec'),
               ('楼梯', 'stair_speed_avg'), ('32分', 'thirtysecond_run_ratio'),
               ('爆发', 'fast_ms_100_ratio'), ('读谱', 'jline_movement_density'),
               ('变速', 'tempo_change_log_density'), ('耐力', 'above_avg_duration_sec'),
               ('高BPM', 'bpm'), ('纵连', 'jack_density'), ('叠键', 'chord_jack_3plus_pairs'),
               ('位移', 'movement_per_second')]
    def tags(feats):
        out = []
        for name, fk in TAG_DIM:
            if feats.get(fk, 0) >= TAG_TH.get(name, 1e9): out.append(name)
        if feats.get('tracks_6plus_sec', 0) / max(feats.get('tracks_active_sec', 1), 0.01) >= TAG_TH.get('定轨', 1): out.append('定轨')
        return out
    def predict(feats_raw, level='IN'):
        feats = dict(feats_raw)
        lv = level.upper()
        if 'AT' in lv: lv = 'AT'
        elif 'IN' in lv: lv = 'IN'
        elif 'HD' in lv: lv = 'HD'
        elif 'EZ' in lv: lv = 'EZ'
        else: lv = 'IN'
        if lv == 'IN':
            for k, d in _ALIGN.items():
                if k in feats: feats[k] = feats[k] - d
        if 'IN_AT' in LV_ORDER and lv in ('IN','AT'): lv = 'IN_AT'
        if lv not in LV_ORDER: lv = LV_ORDER[-1]
        vec = [0.0]*len(LV_ORDER); vec[LV_ORDER.index(lv)] = 1.0
        x = np.array([[feats.get(n,0) for n in FN] + vec])
        p_gb = float(gb.predict(scaler.transform(x))[0])
        mf3 = feats_raw.get('multi_finger_3plus_events', 0); dens = feats_raw.get('above_avg_density_mean', 0)
        ml = feats_raw.get('multi_line_sim_events', 0); wmf = feats_raw.get('weighted_mf_score_per_sec', 0)
        # v11.2 逻辑
        if version == 'v11.2':
            if mf3 >= 30 and ml >= 100: mf_scale, dens_s = 0.45, 0.85
            elif mf3 >= 30: mf_scale, dens_s = (0.70 if dens >= 9.5 else 0.50), 1.0
            else: mf_scale, dens_s = (1.0 if mf3 <= 5 else 0.8), 1.0
            df_stack = mf3 <= 5 and wmf >= 15.0
            eff_scale = 1.0 if mf3 >= 30 else (1.00 if df_stack else (1.50 if mf3 <= 5 else 1.0))
            wmf_scale = 0.60 if df_stack else 1.0
            extreme_scale = 1.0
            stack_scale = 1.0
        else:  # v11.8b
            if mf3 >= 30 and ml >= 100: mf_scale, dens_s = 0.45, 0.85
            elif mf3 >= 30: mf_scale, dens_s = (0.70 if dens >= 9.5 else 0.50), 1.0
            else: mf_scale, dens_s = (1.0 if mf3 <= 5 else 0.8), 1.0
            if mf3 <= 5:
                _sw = min(max((wmf - 12.0) / 6.0, 0.0), 1.0)
                eff_scale = 1.0 if dens >= 10.0 else 1.5 - 0.5 * _sw
                wmf_scale = 1.0 - 0.4 * _sw
                extreme_scale = 1.3
            elif mf3 >= 30:
                eff_scale, wmf_scale = 1.0, 1.0
                extreme_scale = 0.7
            else:
                eff_scale, wmf_scale = 1.0, 1.0
                extreme_scale = 1.0
            ts = tags(feats)
            high = {'叠键', '多押', '变速', '位移'}
            stack_scale = 0.92 if sum(1 for t in ts if t in high) >= 2 else 1.0
        total = 0.0; cd_ = CAPS.get('_default', None)
        for fname, bl, co in FLAT:
            v = feats.get(fname, 0); pv = P95.get(fname, 0)
            t = max(pv*0.55, bl*0.5)
            if v <= t: continue
            e = v/t - 1.0
            c = CAPS.get(fname, cd_)
            if c is not None and e > c: e = c
            co2 = co
            if fname in MF_FEATS: co2 = co * mf_scale
            elif fname in EFF_FEATS: co2 = co * eff_scale
            if fname in DENS_FEATS and mf3 >= 30 and ml >= 100: co2 = co * dens_s
            if fname == 'weighted_mf_score_per_sec': co2 = co * wmf_scale
            if version == 'v11.8b':
                if fname in EXTREME_FEATS: co2 = co * extreme_scale
                co2 = co2 * stack_scale
            x_ = co2 * (e**0.70)
            p99 = max(P99.get(fname,0), bl*0.5)
            if v > p99:
                pe = v/p99 - 1.0
                if c is not None and pe > c: pe = c
                x_ += co2*max(0,pe)**0.70*0.5
            total += x_
        pred = p_gb + total
        act = feats_raw.get('tracks_active_sec', 0)
        if act > 0:
            r4 = feats_raw.get('tracks_4plus_sec', 0)/act; r5 = feats_raw.get('tracks_5plus_sec', 0)/act; r6 = feats_raw.get('tracks_6plus_sec', 0)/act
            pred += 0.15*min(r4,0.8) + 0.55*min(r5,0.4) + 1.0*min(r6,0.15)
        for lo, hi, adj in calib:
            if lo < pred <= hi: pred -= adj; break
        return pred
    return predict

m2 = load_model('6dim_model_v11_2.pkl')
m8 = load_model('6dim_model_v11_7b.pkl')
p2 = make_predictor(m2, [(14,15,0.30),(15,16,0.18),(16,17,0.05)], 'v11.2')
p8 = make_predictor(m8, [(14,15,0.51),(15,16,0.36),(16,17,0.16)], 'v11.8b')

with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([r['diff'] for r in ranked])
ps2 = np.array([p2(r['feats'], r['level']) for r in ranked])
ps8 = np.array([p8(r['feats'], r['level']) for r in ranked])
e2 = ps2 - ds; e8 = ps8 - ds

print('===== v11.2 vs v11.8b 上架谱预测对比 =====')
print(f'{"指标":<14}{"v11.2":>10}{"v11.8b":>10}{"变化":>10}')
print(f'{"MAE":<14}{np.abs(e2).mean():>10.3f}{np.abs(e8).mean():>10.3f}{np.abs(e8).mean()-np.abs(e2).mean():>+10.3f}')
print(f'{"bias":<14}{e2.mean():>10.3f}{e8.mean():>10.3f}{e8.mean()-e2.mean():>+10.3f}')
print(f'{"RMSE":<14}{np.sqrt((e2**2).mean()):>10.3f}{np.sqrt((e8**2).mean()):>10.3f}{np.sqrt((e8**2).mean())-np.sqrt((e2**2).mean()):>+10.3f}')
print(f'{"rho":<14}{spearmanr(ds, ps2)[0]:>10.3f}{spearmanr(ds, ps8)[0]:>10.3f}{spearmanr(ds, ps8)[0]-spearmanr(ds, ps2)[0]:>+10.3f}')
print(f'{"|err|>1":<14}{(np.abs(e2)>1).sum():>10d}{(np.abs(e8)>1).sum():>10d}{-((np.abs(e8)>1).sum()-(np.abs(e2)>1).sum()):>+10d}')
print(f'{"|err|>2":<14}{(np.abs(e2)>2).sum():>10d}{(np.abs(e8)>2).sum():>10d}{-((np.abs(e8)>2).sum()-(np.abs(e2)>2).sum()):>+10d}')
print('\n按社区定数段 (bias/MAE):')
for lo, hi, tag in [(10,14,'<14'),(14,15,'14-15'),(15,16,'15-16'),(16,17,'16-17'),(17,99,'>=17')]:
    mk = np.where((ds >= lo) & (ds < hi))[0]
    if len(mk):
        print(f'  [{tag}] n={len(mk):3d} | v11.2: {e2[mk].mean():+.3f}/{np.abs(e2[mk]).mean():.3f} | v11.8b: {e8[mk].mean():+.3f}/{np.abs(e8[mk]).mean():.3f}')
mf = np.array([r['feats'].get('multi_finger_3plus_events', 0) for r in ranked])
print('\n多指/双指:')
for lbl, cond in [('多指', mf>=30), ('双指', mf<=5), ('混合', (mf>5)&(mf<30))]:
    mk = np.where(cond)[0]
    if len(mk):
        print(f'  {lbl}: n={len(mk)} | v11.2: {e2[mk].mean():+.3f}/{np.abs(e2[mk]).mean():.3f} | v11.8b: {e8[mk].mean():+.3f}/{np.abs(e8[mk]).mean():.3f}')
# 保存
import csv as _csv
with open(os.path.join(_ROOT, 'data', 'phira', 'v112_vs_v118b.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = _csv.writer(f)
    w.writerow(['name', 'level', 'diff', 'pred_v112', 'pred_v118b', 'err_v112', 'err_v118b'])
    for i, r in enumerate(ranked):
        w.writerow([r['name'], r['level'], r['diff'], round(ps2[i],3), round(ps8[i],3), round(e2[i],3), round(e8[i],3)])
print('\n已保存: v112_vs_v118b.csv')
print('DONE')
