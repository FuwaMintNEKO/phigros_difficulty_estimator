# -*- coding: utf-8 -*-
"""t2 方案B验证 v2: above_avg_density_mean 改为按有效单指计算 (官谱982精确重算)
直接遍历缓存 official, 用 (name, level) 精确匹配谱面文件
"""
import os, sys, pickle
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
_ROOT = r'D:\\Trae项目\\新建文件夹\\phigros_difficulty_estimator'
sys.path.insert(0, _ROOT)

from data_loader import find_chart_files, load_chart_json
from feature_extractor import collect_all_notes, time_to_seconds, _compute_duration_sec

def eff_density_features(cd):
    all_notes, judge_lines, bpm_timeline = collect_all_notes(cd)
    if not all_notes:
        return None
    fallback_bpm = judge_lines[0].get('bpm', 120.0) if judge_lines else 120.0
    bpm = bpm_timeline[0][1] if bpm_timeline else fallback_bpm
    times = np.array([n['time'] for n in all_notes])
    types = np.array([n['type'] for n in all_notes])
    tap_mask = types == 1
    hold_mask = types == 3
    core_mask = tap_mask | hold_mask

    dt = float(times[-1]) if len(times) > 0 else 0
    duration_beats = dt / 32
    duration_sec = _compute_duration_sec(bpm_timeline, duration_beats)

    n_notes = len(all_notes)
    core_n = int(np.sum(core_mask))
    rest_gap_threshold = 1.0
    all_t_sec = np.array([time_to_seconds(t, max(n.get('bpm', bpm), 1.0), bpm_timeline) for t, n in zip(times, all_notes)])
    all_t_sec.sort()
    if n_notes > 1:
        gaps = np.diff(all_t_sec)
        big_gaps = gaps[gaps > rest_gap_threshold]
        rest_duration = float(np.sum(big_gaps))
        half = rest_gap_threshold / 2.0
        active = 0.0
        cur_end = None
        for t in all_t_sec:
            s, e = t - half, t + half
            if cur_end is None or s > cur_end:
                active += e - s
                cur_end = e
            elif e > cur_end:
                active += e - cur_end
                cur_end = e
        real_active = max(active, 0.01)
    else:
        real_active = max(duration_sec, 0.01)
    rcnps = core_n / real_active

    # ===== eff 计算 (复制 773-801) =====
    core_idx = np.where(core_mask)[0]
    core_times = times[core_mask]
    if len(core_times) > 5:
        core_t_sec = np.array([time_to_seconds(t, max(all_notes[idx].get('bpm', bpm), 1.0), bpm_timeline)
                               for idx, t in zip(core_idx, core_times)])
        order = np.argsort(core_t_sec)
        cts_sorted = core_t_sec[order]
        ctk_sorted = core_times[order]
        left = 0
        max_eff = 0
        eff_vals = []
        for right in range(len(cts_sorted)):
            while cts_sorted[right] - cts_sorted[left] > 1.0:
                left += 1
            seg = ctk_sorted[left:right + 1]
            if len(seg) >= 2:
                eff = 1 + int(np.sum(np.diff(seg) >= 1))
            else:
                eff = int(len(seg))
            max_eff = max(max_eff, eff)
            eff_vals.append(eff)
        eff_peak = int(max_eff)
        eff_avg = float(np.mean(eff_vals))
    else:
        eff_peak = 0
        eff_avg = 0.0

    # ===== above_avg: 原始 vs eff 版 =====
    above_avg_orig = rcnps
    above_avg_eff = rcnps
    if len(core_times) > 5:
        t_arr = np.sort(core_t_sec)
        left = 0
        above_windows = []
        for right in range(len(t_arr)):
            while t_arr[right] - t_arr[left] > 1.0:
                left += 1
            window_tps = right - left + 1
            if window_tps >= rcnps:
                above_windows.append(window_tps)
        if above_windows:
            above_avg_orig = float(np.mean(above_windows))
        # eff 版: 同一滑动窗口, 有效击打数
        left2 = 0
        eff_counts = []
        for right in range(len(cts_sorted)):
            while cts_sorted[right] - cts_sorted[left2] > 1.0:
                left2 += 1
            seg = ctk_sorted[left2:right+1]
            eff_cnt = 1 + int(np.sum(np.diff(seg) >= 1)) if len(seg) >= 2 else int(len(seg))
            eff_counts.append(eff_cnt)
        thresh_eff = max(eff_avg, 0.01)
        above_eff = [c for c in eff_counts if c >= thresh_eff]
        if above_eff:
            above_avg_eff = float(np.mean(above_eff))

    return {
        'rcnps': rcnps, 'eff_avg': eff_avg, 'eff_peak': eff_peak,
        'above_avg_orig': above_avg_orig, 'above_avg_eff': above_avg_eff,
    }

chart_files = find_chart_files(os.path.join(_ROOT, 'data', 'chart'))
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)

rows = []
skipped = 0
for o in cache['official']:
    fn, lv = o['name'], o['level']
    info = chart_files.get(fn)
    if not info or lv not in info['levels']:
        skipped += 1
        continue
    try:
        cd = load_chart_json(info['levels'][lv])
        r = eff_density_features(cd)
        if not r:
            skipped += 1
            continue
        f = o['feats']
        rows.append({
            'name': fn, 'level': lv, 'diff': float(o['diff']),
            'cached_dens': f.get('above_avg_density_mean', 0),
            'cached_effa': f.get('eff_avg_tps_1s', 0),
            'cached_effp': f.get('eff_peak_tps_1s', 0),
            'cached_rcnps': f.get('real_core_notes_per_second', 0),
            **r,
        })
    except Exception:
        skipped += 1

print(f'重算官谱数: {len(rows)} (跳过 {skipped})')

def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12: return float('nan')
    return float(np.corrcoef(x, y)[0, 1])

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3: return float('nan')
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12); ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.dot(rx, ry) / len(rx))

c_effa = np.array([r['cached_effa'] for r in rows])
m_effa = np.array([r['eff_avg'] for r in rows])
c_effp = np.array([r['cached_effp'] for r in rows])
m_effp = np.array([r['eff_peak'] for r in rows])
c_dens = np.array([r['cached_dens'] for r in rows])
m_oa = np.array([r['above_avg_orig'] for r in rows])
c_rcnps = np.array([r['cached_rcnps'] for r in rows])
m_rcnps = np.array([r['rcnps'] for r in rows])

print()
print('===== 复制逻辑交叉验证 (vs 缓存) =====')
print(f'  eff_avg : max diff={np.max(np.abs(m_effa-c_effa)):.4f} corr={pearson(m_effa, c_effa):.4f}')
print(f'  eff_peak: max diff={np.max(np.abs(m_effp-c_effp)):.0f} 一致率={np.mean(m_effp==c_effp)*100:.1f}%')
print(f'  above_avg原始: max diff={np.max(np.abs(m_oa-c_dens)):.4f} corr={pearson(m_oa, c_dens):.4f}')
print(f'  rcnps   : max diff={np.max(np.abs(m_rcnps-c_rcnps)):.4f} corr={pearson(m_rcnps, c_rcnps):.4f}')

D = np.array([r['diff'] for r in rows])
dens_orig = np.array([r['above_avg_orig'] for r in rows])
dens_eff = np.array([r['above_avg_eff'] for r in rows])
print()
print('===== 方案B: above_avg_density_mean 原始 vs 有效单指版 =====')
print(f'  原始版: mean={dens_orig.mean():.2f} P50={np.median(dens_orig):.2f} P90={np.percentile(dens_orig,90):.2f}')
print(f'         与diff corr: P={pearson(dens_orig,D):.4f} S={spearman(dens_orig,D):.4f}')
print(f'  eff版 : mean={dens_eff.mean():.2f} P50={np.median(dens_eff):.2f} P90={np.percentile(dens_eff,90):.2f}')
print(f'         与diff corr: P={pearson(dens_eff,D):.4f} S={spearman(dens_eff,D):.4f}')
valid_ratio = dens_eff[dens_orig>0.5] / dens_orig[dens_orig>0.5]
print(f'  eff版/原始版 (dens>0.5): mean={np.mean(valid_ratio):.3f} P10={np.percentile(valid_ratio,10):.3f} P90={np.percentile(valid_ratio,90):.3f}')

print()
print('===== 按定数段 原始 vs eff版 =====')
bins = [('<13', 0, 13), ('13-14', 13, 14), ('14-15', 14, 15), ('15-16', 15, 16), ('16-17', 16, 17), ('>=17', 17, 99)]
for name, lo, hi in bins:
    sel = [r for r in rows if lo <= r['diff'] < hi]
    if not sel: continue
    o = np.array([r['above_avg_orig'] for r in sel])
    e = np.array([r['above_avg_eff'] for r in sel])
    ratio = e[np.maximum(o,0.5)>0.5] / np.maximum(o[np.maximum(o,0.5)>0.5],0.5)
    print(f'  {name:<6} n={len(sel):<4} 原始={o.mean():.2f}  eff={e.mean():.2f}  均值比={np.mean(ratio):.3f}')

print()
print('===== eff版降幅最大的 12 张 (多押撑密度) =====')
def ratio_of(r):
    return r['above_avg_eff']/max(r['above_avg_orig'],0.5)
for r in sorted(rows, key=ratio_of)[:12]:
    print(f'  {r["name"][:36]:<38} {r["level"]} diff={r["diff"]:>5.1f} 原始={r["above_avg_orig"]:>5.1f} eff={r["above_avg_eff"]:>5.2f} 比={ratio_of(r):.3f} eff_avg={r["eff_avg"]:.2f}')

with open(os.path.join(_ROOT, 'tools', '_tmp_planB_results.pkl'), 'wb') as f:
    pickle.dump(rows, f)
print()
print('已保存 tools/_tmp_planB_results.pkl')
