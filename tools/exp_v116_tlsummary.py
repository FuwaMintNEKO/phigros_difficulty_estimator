# -*- coding: utf-8 -*-
"""v11.6 批量时间线摘要: 玩家视角结构特征
每谱输出: peak_nps / rest_ratio / tail_peak / chord_peak / chord_heavy_ratio / 结构类型
"""
import os, sys, io, numpy as np, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from data_loader import load_chart_json, find_chart_files
from feature_extractor import collect_all_notes, time_to_seconds

def timeline_summary(chart_data, win=8.0):
    all_notes, judge_lines, bpm_timeline = collect_all_notes(chart_data)
    if not all_notes: return None
    n = len(all_notes)
    times = np.array([nd['time'] for nd in all_notes])
    types = np.array([nd['type'] for nd in all_notes])
    bpms = np.array([nd['bpm'] for nd in all_notes])
    secs = np.array([time_to_seconds(t, max(b, 1.0), bpm_timeline) for t, b in zip(times, bpms)])
    dur = secs[-1] if len(secs) else 0
    nw = int(np.ceil(dur / win))
    nps_list, chord_list = [], []
    for w in range(nw):
        lo, hi = w*win, (w+1)*win
        m = (secs >= lo) & (secs < hi)
        if not m.any():
            nps_list.append(0); chord_list.append(0); continue
        ts = times[m]; ty = types[m]
        core = (ty == 1) | (ty == 3)
        core_t = np.sort(ts[core])
        groups = 1 + np.sum(np.diff(core_t) > 0.02) if len(core_t) > 0 else 0
        nps_list.append(groups / win)
        chord_list.append(int(np.sum(np.diff(ts) < 0.02)) + 1 if len(ts) > 0 else 0)
    nps_arr = np.array(nps_list); ch_arr = np.array(chord_list)
    tail_start = int(nw * 0.75)
    return {
        'dur': round(dur, 1), 'windows': nw,
        'peak_nps': round(float(nps_arr.max()), 2),
        'mean_nps': round(float(nps_arr.mean()), 2),
        'rest_ratio': round(float(np.mean(nps_arr < 2.0)), 3),   # 休息窗口占比
        'tail_peak': round(float(nps_arr[tail_start:].max()), 2) if tail_start < nw else 0,  # 尾杀峰值
        'chord_peak': int(ch_arr.max()),
        'chord_mean': round(float(ch_arr.mean()), 1),
        'chord_heavy_ratio': round(float(np.mean(ch_arr >= 20)), 3),  # 多押>=20窗口占比
        'hard_ratio': round(float(np.mean(nps_arr >= 6.0)), 3),  # 高密度窗口占比
    }

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'official'
    if mode == 'official':
        chart_dir = os.path.join(_ROOT, 'data', 'chart')
        items = []
        for fn, info in find_chart_files(chart_dir).items():
            for lv in ['IN', 'AT']:
                if lv in info['levels']:
                    try:
                        cd = load_chart_json(info['levels'][lv])
                        s = timeline_summary(cd)
                        if s: items.append((fn, lv, s))
                    except Exception:
                        pass
                    break
        # 输出 15+ 谱 (按官方定数? 这里只按文件名, 排序按 chord_heavy)
        items.sort(key=lambda x: -x[2]['chord_heavy_ratio'])
        print(f'{"谱面":<36}{"难度":<4}{"dur":>6}{"peak":>5}{"mean":>5}{"rest%":>6}{"tail":>5}{"chPK":>5}{"chMe":>5}{"chHv%":>6}{"hard%":>6}')
        for fn, lv, s in items[:40]:
            print(f'{fn[:34]:<36}{lv:<4}{s["dur"]:>6.0f}{s["peak_nps"]:>5.1f}{s["mean_nps"]:>5.1f}{s["rest_ratio"]*100:>6.0f}{s["tail_peak"]:>5.1f}{s["chord_peak"]:>5}{s["chord_mean"]:>5.1f}{s["chord_heavy_ratio"]*100:>6.0f}{s["hard_ratio"]*100:>6.0f}')
        print('DONE')
