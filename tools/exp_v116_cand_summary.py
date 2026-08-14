# -*- coding: utf-8 -*-
"""社区候选谱批量时间线摘要 vs 官谱对照
"""
import os, sys, io, numpy as np, json, csv, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from unified_parser import load_chart_from_bytes
from feature_extractor import collect_all_notes, time_to_seconds

def tl_summary(cd, win=8.0):
    all_notes, judge_lines, bpm_timeline = collect_all_notes(cd)
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
        'rest_ratio': round(float(np.mean(nps_arr < 2.0)), 3),
        'tail_peak': round(float(nps_arr[tail_start:].max()), 2) if tail_start < nw else 0,
        'chord_peak': int(ch_arr.max()),
        'chord_mean': round(float(ch_arr.mean()), 1),
        'chord_heavy_ratio': round(float(np.mean(ch_arr >= 20)), 3),
        'hard_ratio': round(float(np.mean(nps_arr >= 6.0)), 3),
    }

# 候选谱: 从 CSV 读取 (多指+低密+被低估 + Cheerio)
cands = []
with open(os.path.join(_ROOT, 'data', 'phira', 'unranked_4star_list.csv'), encoding='utf-8-sig') as f:
    rd = csv.DictReader(f)
    for r_ in rd:
        try:
            d = float(r_['difficulty']) if r_['difficulty'] else None
            if d is None or not (16.0 <= d < 25.0): continue
            if float(r_['rating']) < 0.9 or int(r_['ratingCount']) < 100: continue
            mf3 = float(r_['mf3']); dens = float(r_['dens'])
            if mf3 >= 30 and dens < 9.5 and (float(r_['pred']) - d) < -0.5:
                cands.append((int(r_['id']), r_['name'], r_['level'], d, float(r_['pred']), mf3, dens))
        except Exception:
            pass
cands.append((58496, 'Cheerio!', 'AT Lv.17', 17.1, 16.47, 97, 8.36))
print(f'候选: {len(cands)}')

JSON_DIR = os.path.join(_ROOT, 'data', 'phira', 'json_unranked_4star')
out = []
for cid, name, lv, d, pred, mf3, dens in cands:
    fp = os.path.join(JSON_DIR, f'{cid}.json')
    if not os.path.exists(fp): continue
    try:
        with open(fp, 'rb') as f:
            cd, raw = load_chart_from_bytes(f.read())
        s = tl_summary(cd)
        if s:
            s.update({'name': name[:22], 'diff': d, 'pred': pred, 'err': round(pred-d, 2), 'mf3': mf3, 'dens': dens})
            out.append(s)
    except Exception:
        pass
out.sort(key=lambda x: x['err'])
print(f'{"谱面":<24}{"社区":>5}{"预测":>6}{"err":>7}{"mf3":>5}{"dens":>5}{"dur":>5}{"pkN":>5}{"mnN":>5}{"rest%":>5}{"tail":>5}{"chPK":>5}{"chMe":>5}{"chHv%":>6}{"hard%":>5}')
for s in out:
    print(f'{s["name"]:<24}{s["diff"]:>5.1f}{s["pred"]:>6.2f}{s["err"]:>+7.2f}{s["mf3"]:>5.0f}{s["dens"]:>5.1f}{s["dur"]:>5.0f}{s["peak_nps"]:>5.1f}{s["mean_nps"]:>5.1f}{s["rest_ratio"]*100:>5.0f}{s["tail_peak"]:>5.1f}{s["chord_peak"]:>5}{s["chord_mean"]:>5.1f}{s["chord_heavy_ratio"]*100:>6.0f}{s["hard_ratio"]*100:>5.0f}')
print('DONE')
