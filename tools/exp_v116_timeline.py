# -*- coding: utf-8 -*-
"""v11.6 玩家视角工具: 谱面时间线难度曲线
按时间窗口输出: 核心nps(仅tap+hold) / 有效单指tps / 多押事件 / 类型占比
用途: 客观化玩家评价("前半简单""休息段多""高潮段难""尾杀")
"""
import os, sys, io, numpy as np, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from data_loader import load_chart_json, find_chart_files
from feature_extractor import collect_all_notes, collect_speed_events, time_to_seconds

def timeline(chart_data, win=8.0):
    all_notes, judge_lines, bpm_timeline = collect_all_notes(chart_data)
    if not all_notes: return None
    n = len(all_notes)
    times = np.array([nd['time'] for nd in all_notes])
    types = np.array([nd['type'] for nd in all_notes])
    bpms = np.array([nd['bpm'] for nd in all_notes])
    positions = np.array([nd.get('positionX', 0) for nd in all_notes])
    # 时间 → 秒 (用BPM timeline)
    secs = np.array([time_to_seconds(t, max(b, 1.0), bpm_timeline) for t, b in zip(times, bpms)])
    dur = secs[-1] if len(secs) else 0
    nw = int(np.ceil(dur / win))
    out = []
    for w in range(nw):
        lo, hi = w*win, (w+1)*win
        m = (secs >= lo) & (secs < hi)
        if not m.any():
            out.append({'t': lo, 'nps': 0, 'eff': 0, 'chords': 0, 'tap%': 0, 'hold%': 0, 'drag%': 0, 'flick%': 0})
            continue
        ts = times[m]; ty = types[m]; ps = positions[m]
        # 核心nps: tap+hold 去同押
        core = (ty == 1) | (ty == 3)
        # 同押去重: 时间差<0.02拍 视为同押
        core_t = ts[core]
        if len(core_t) > 0:
            groups = 1 + np.sum(np.diff(core_t) > 0.02)
        else:
            groups = 0
        nps = groups / win
        # 有效单指: 同押组内只算1
        eff = groups / win
        chords = int(np.sum(np.diff(ts) < 0.02)) + 1 if len(ts) > 0 else 0
        n_tap = int(np.sum(ty == 1)); n_hold = int(np.sum(ty == 3)); n_drag = int(np.sum(ty == 2)); n_fl = int(np.sum(ty == 4))
        tot = len(ty)
        out.append({'t': round(lo,1), 'nps': round(nps,2), 'eff': round(eff,2), 'chords': chords,
                    'tap%': round(100*n_tap/max(tot,1)), 'hold%': round(100*n_hold/max(tot,1)),
                    'drag%': round(100*n_drag/max(tot,1)), 'flick%': round(100*n_fl/max(tot,1))})
    return out, dur

def render(rows, dur, title=''):
    print(f'\n===== {title} (全长{dur:.0f}s, 窗口8s) =====')
    print('  时间  nps(核心)  eff(单指) 多押  tap% hold% drag%')
    for r in rows:
        bar = '#' * min(int(r['nps'] * 2), 60)
        print(f'  {r["t"]:>5.0f}s  {r["nps"]:>6.2f}   {r["eff"]:>6.2f}  {r["chords"]:>3}  {r["tap%"]:>3}  {r["hold%"]:>4}  {r["drag%"]:>4}')
        if r['nps'] >= 4: print(f'         ↑高潮 {bar}')

if __name__ == '__main__':
    targets = sys.argv[1:]
    # 社区谱 (json_unranked_4star)
    from unified_parser import load_chart_from_bytes
    json_dir = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star')
    if targets and targets[0] == '--rpe':
        ids = targets[1:]
        for cid in ids:
            fp = os.path.join(json_dir, f'{cid}.json')
            if not os.path.exists(fp):
                print(f'{cid} 不存在'); continue
            with open(fp, 'rb') as f:
                cd, raw = load_chart_from_bytes(f.read())
            rows, dur = timeline(cd)
            render(rows, dur, f'社区谱 id={cid}')
        sys.exit(0)
    chart_dir = os.path.join(_ROOT, 'data', 'chart')
    for kw in targets:
        for fn, info in find_chart_files(chart_dir).items():
            if kw.lower() in fn.lower():
                for lv in ['IN', 'AT']:
                    if lv in info['levels']:
                        try:
                            cd = load_chart_json(info['levels'][lv])
                            rows, dur = timeline(cd)
                            render(rows, dur, f'{fn} {lv}')
                        except Exception as e:
                            print(f'{fn} {lv} ERR: {e}')
                        break
