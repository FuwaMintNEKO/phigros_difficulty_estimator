"""v2: 更精准的节奏分析 — 用实际beat值聚类, 不是宽松snap"""
import sys, os, json, numpy as np, math, re
from collections import Counter
sys.path.insert(0, '.')
import io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from feature_extractor import collect_all_notes, time_to_seconds, extract_features
from unified_parser import load_chart_from_bytes

def compute_intervals_beat(all_notes):
    """返回 [(分音数, 描述, beat值)], BPM已归一化"""
    results = []
    n = len(all_notes)
    for i in range(n):
        n0 = all_notes[i]
        line0 = n0.get('judge_line_idx', 0)
        t0_sec = time_to_seconds(n0['time'], max(n0.get('bpm', 120), 1.0))
        
        for j in range(i + 1, min(i + 50, n)):  # 最多看50个以后的
            nj = all_notes[j]
            if nj.get('judge_line_idx', 0) != line0:
                continue
            tj_sec = time_to_seconds(nj['time'], max(nj.get('bpm', 120), 1.0))
            gap_sec = tj_sec - t0_sec
            if gap_sec <= 0.005:  # skip too close
                continue
            avg_bpm = (n0.get('bpm', 120) + nj.get('bpm', 120)) / 2
            beats = gap_sec * avg_bpm / 60.0
            
            # 判断分音类型 (严格tolerance)
            if beats > 1.5:  # 超过1.5拍, 不再看
                break
            
            # 严格匹配
            matched = None
            for frac, (name, target) in {
                2: ('8分', 0.5), 3: ('8分三连', 1/3), 4: ('16分', 0.25),
                5: ('5连', 0.2), 6: ('16分三连', 1/6), 7: ('7连', 1/7),
                8: ('32分', 0.125), 9: ('9连', 1/9), 12: ('12连', 1/12),
                14: ('14连', 1/14), 16: ('64分', 0.0625), 24: ('24分', 1/24),
                28: ('28分', 1/28), 32: ('128分', 0.03125),
            }.items():
                if abs(beats - target) / max(target, 0.001) < 0.12:  # 12% tolerance
                    matched = (frac, name, beats, gap_sec)
                    break
            
            if matched:
                results.append(matched)
                break  # 只取最近的一个
            elif beats > 0.02:
                # 不匹配任何标准类型 → 不规则节奏
                results.append((0, 'irregular', beats, gap_sec))
                break
    return results

def compute_chord_features(all_notes):
    """多押分析: 按time分组"""
    time_groups = {}
    for note in all_notes:
        # 分组: 同一秒内0.03秒窗口算同时 (考虑到bpm变化)
        t_sec = time_to_seconds(note['time'], max(note.get('bpm', 120), 1.0))
        t_bin = round(t_sec, 2)  # 10ms bin
        if t_bin not in time_groups:
            time_groups[t_bin] = []
        time_groups[t_bin].append(note)
    
    chords = [(t, len(g)) for t, g in time_groups.items() if len(g) >= 2]
    if not chords:
        return {'max_chord': 1, 'avg_chord': 1, 'mf_weight_score': 0, 'poly_density': 0}
    
    sizes = [s for _, s in chords]
    # 梯形权重: 2押=1, 3押=3, 4押=6, 5押=10, 6押=15
    mf_score = sum(s * (s - 1) / 2 for s in sizes)
    
    return {
        'max_chord': max(sizes),
        'avg_chord': np.mean(sizes),
        'mf_weight_score': mf_score,
        'poly_count': len(chords),
        'poly_density': len(chords) / (len(chords) + sum(1 for t, g in time_groups.items() if len(g) == 1) + 1),
    }

# ── 收集数据 ──
test_dir = r'C:\Users\NaNK\Downloads'
charts = []
for fn in os.listdir(test_dir):
    if not fn.endswith('.json'): continue
    if '_2xBPM' in fn or '_2x' in fn: continue
    fp = os.path.join(test_dir, fn)
    if os.path.getsize(fp) < 100: continue
    try:
        rating = None
        for m in re.finditer(r'\((\d+\.?\d*)\)', fn):
            val = float(m.group(1))
            if 5 <= val <= 20: rating = val; break
        if rating is None: continue
        with open(fp, 'rb') as f: raw = f.read()
        data, _ = load_chart_from_bytes(raw)
        feats = extract_features(data)
        all_notes, _, _ = collect_all_notes(data)
        charts.append((fn, feats, all_notes, rating))
    except: pass

print(f'测试谱: {len(charts)}个\n')

# ── 计算 ──
new_feats = []
rhythm_maps = []
for fn, old_feats, all_notes, rating in charts:
    dur = old_feats.get('duration_sec', 1)
    
    # 间隔分析
    intervals = compute_intervals_beat(all_notes)
    
    # 节奏类型计数
    rhythm_types = Counter()
    fast_16th_count = 0
    fast_32nd_count = 0
    irregular_count = 0
    
    for frac, name, beats, gap_sec in intervals:
        if frac >= 2:
            rhythm_types[name] += 1
        if frac >= 4:
            fast_16th_count += 1
        if frac >= 8:
            fast_32nd_count += 1
        if frac == 0:
            irregular_count += 1
    
    # 多押
    chord = compute_chord_features(all_notes)
    
    nf = {
        'fast16th_dens': fast_16th_count / max(dur, 0.01),
        'fast32nd_dens': fast_32nd_count / max(dur, 0.01),
        'irregular_dens': irregular_count / max(dur, 0.01),
        'rhythm_type_count': len(rhythm_types),
        'rhythm_entropy': -(sum(c/max(sum(rhythm_types.values()),1) * math.log(max(c/max(sum(rhythm_types.values()),1), 0.001))
                             for c in rhythm_types.values()) / math.log(2)) if rhythm_types else 0,
        'max_chord_new': chord['max_chord'],
        'avg_chord_new': chord['avg_chord'],
        'mf_weight_new': chord['mf_weight_score'] / max(dur, 0.01),
        'poly_density': chord['poly_density'],
    }
    new_feats.append(nf)
    rhythm_maps.append(rhythm_types)

# ── 相关性 ──
print('=' * 80)
print('  新特征 vs 难度 相关性')
print('=' * 80)
ratings_arr = [r for _, _, _, r in charts]
for key in ['fast16th_dens', 'fast32nd_dens', 'irregular_dens', 'rhythm_type_count',
            'rhythm_entropy', 'max_chord_new', 'avg_chord_new', 'mf_weight_new', 'poly_density']:
    vals = [nf[key] for nf in new_feats]
    corr = np.corrcoef(vals, ratings_arr)[0, 1]
    print(f'  {key:<25} corr={corr:+.4f}  range=[{min(vals):.2f}~{max(vals):.2f}]')

# ── 详细对比 ──
print(f'\n{"="*130}')
print(f'  谱面对比')
print(f'{"="*130}')
header = f'{"谱面":<14} {"定数":>5} {"旧配置":>7} {"16th/s":>7} {"32nd/s":>7} {"irreg":>6} {"rTypes":>6} {"rEntr":>6} {"maxC":>5} {"mf/sec":>7} {"节奏类型":>30}'
print(header)
print('-' * 130)

# 旧配置近似
def old_cfg(feats):
    return (feats.get('stair_density',0)*0.05 + feats.get('chord_size_entropy',0)*0.5
            + feats.get('multi_finger_3plus_events',0)*0.002 + feats.get('chord_alternation_rate',0)*0.05
            + feats.get('weighted_mf_score_per_sec',0)*0.01)

for (fn, old_feats, _, rating), nf, rhythms in zip(charts, new_feats, rhythm_maps):
    top_rhythms = ','.join(f'{k}({v})' for k, v in rhythms.most_common(4))
    print(f'{fn[:14]:<14} {rating:>5.1f} {old_cfg(old_feats):>7.2f} {nf["fast16th_dens"]:>7.2f} '
          f'{nf["fast32nd_dens"]:>7.2f} {nf["irregular_dens"]:>6.2f} {nf["rhythm_type_count"]:>6.0f} '
          f'{nf["rhythm_entropy"]:>6.2f} {nf["max_chord_new"]:>5.0f} {nf["mf_weight_new"]:>7.2f} '
          f'{top_rhythms:>30}')

# ── 重点对比: FinalEndGame vs Apollo ──
print(f'\n{"="*80}')
print('  键盘谱(FinalEndGame, 胧月) vs 正常谱(Apollo, 怪文書) 关键差异')
print(f'{"="*80}')
for fn, old_feats, all_notes, rating in charts:
    if not any(k in fn for k in ['Final EndGame', '朧月', 'Apollo', '怪文書', '恋ひ恋ふ縁']):
        continue
    nf = new_feats[charts.index((fn, old_feats, all_notes, rating))]
    rhythms = rhythm_maps[charts.index((fn, old_feats, all_notes, rating))]
    
    has_hold = feats.get('hold_count', 0)
    
    print(f'\n  {fn[:35]}  rating={rating}')
    print(f'    16th/s={nf["fast16th_dens"]:.1f}  32nd/s={nf["fast32nd_dens"]:.1f}  '
          f'irreg/s={nf["irregular_dens"]:.1f}  rTypes={nf["rhythm_type_count"]}')
    print(f'    maxChord={nf["max_chord_new"]}  avgChord={nf["avg_chord_new"]:.2f}  '
          f'mf/sec={nf["mf_weight_new"]:.1f}  polyDens={nf["poly_density"]:.2f}')
    print(f'    旧特征: stairs={old_feats.get("stair_density",0):.1f}  '
          f'mf3+events={old_feats.get("multi_finger_3plus_events",0)}  '
          f'chord_ent={old_feats.get("chord_size_entropy",0):.2f}')
    print(f'    节奏分布: {dict(rhythms.most_common(6))}')
