import numpy as np
from collections import defaultdict


NOTE_TAP = 1
NOTE_DRAG = 2
NOTE_HOLD = 3
NOTE_FLICK = 4


def _parse_bpm_timeline(chart_data):
    """从 BPMList 构建 BPM 时间线 [(start_beat, bpm), ...] 按起始拍排序
    兼容两种startTime格式:
      - RPE格式: [measure, beat, division] (list)
      - 标准格式: float (已是beats)
    """
    bpm_list = chart_data.get('BPMList', [])
    timeline = []
    if bpm_list:
        for entry in bpm_list:
            st = entry.get('startTime', [0, 0, 1])
            if isinstance(st, (list, tuple)):
                if len(st) >= 3:
                    start_beat = st[0] + (st[1] / max(st[2], 1))
                else:
                    start_beat = st[0]
            else:
                start_beat = float(st) if st else 0.0
            timeline.append((start_beat, entry['bpm']))
        timeline.sort(key=lambda x: x[0])
    return timeline


def _resolve_bpm_at_beat(ticks, bpm_timeline, fallback_bpm):
    """查表获取指定ticks位置的 BPM"""
    if not bpm_timeline:
        return fallback_bpm
    beat_time = ticks / 32  # ticks → beats
    bpm = bpm_timeline[0][1]
    for start_beat, b in bpm_timeline:
        if start_beat <= beat_time:
            bpm = b
        else:
            break
    return bpm


def _compute_duration_sec(bpm_timeline, total_beats):
    """用 BPM 时间线计算真实总时长(秒)，total_beats=谱面总拍数"""
    total_sec = 0.0
    for i, (seg_start, bpm_val) in enumerate(bpm_timeline):
        seg_end = bpm_timeline[i+1][0] if i+1 < len(bpm_timeline) else total_beats
        seg_beats = max(0, min(seg_end, total_beats) - max(seg_start, 0))
        total_sec += seg_beats / bpm_val * 60  # beats ÷ bpm × 60 = 秒
    return total_sec


def _normalize_time_st(note):
    """统一 time 字段: startTime[beats,num,den] → ticks(1/32拍); 无 startTime 则取 time(已是ticks)"""
    st = note.get('startTime', None)
    if st is not None and isinstance(st, (list, tuple)):
        beat = st[0] + (st[1] / max(st[2], 1)) if len(st) >= 3 else st[0]
        return beat * 32  # beats → ticks
    return note.get('time', 0)  # 官谱的 time 字段已是ticks格式

def _beat_from_ticks(ticks):
    """ticks → beats 转换"""
    return ticks / 32


def collect_all_notes(chart_data):
    all_notes = []
    judge_lines = chart_data.get('judgeLineList', [])

    # 构建 BPM 时间线（从 BPMList，如果有的话）
    bpm_timeline = _parse_bpm_timeline(chart_data)
    fallback_bpm = judge_lines[0].get('bpm', 120.0) if judge_lines else 120.0
    has_bpmlist = len(bpm_timeline) > 0

    # 收集所有notes，使用正确的per-line BPM
    line_note_counts = defaultdict(int)
    for li, line in enumerate(judge_lines):
        line_bpm_val = line.get('bpm', fallback_bpm)
        for note in line.get('notes', []) + line.get('notesAbove', []) + line.get('notesBelow', []):
            note_copy = dict(note)
            beat_time = _normalize_time_st(note_copy)
            note_copy['time'] = beat_time
            if has_bpmlist:
                note_copy['bpm'] = _resolve_bpm_at_beat(beat_time, bpm_timeline, fallback_bpm)
            else:
                note_copy['bpm'] = line_bpm_val  # 每条线用自己的BPM
            note_copy['judge_line_idx'] = li
            all_notes.append(note_copy)
            line_note_counts[li] += 1
    all_notes.sort(key=lambda x: x['time'])

    # 对于无BPMList的多BPM谱，构造加权BPM时间线（用于duration等计算）
    if not has_bpmlist and len(judge_lines) > 0:
        weighted_bpm = fallback_bpm
        total_n = len(all_notes)
        if total_n > 0:
            weighted_sum = 0.0
            for li, line in enumerate(judge_lines):
                lb = line.get('bpm', fallback_bpm)
                weighted_sum += lb * line_note_counts.get(li, 0)
            weighted_bpm = weighted_sum / total_n
        bpm_timeline = [(0, weighted_bpm)]

    return all_notes, judge_lines, bpm_timeline


def time_to_seconds(time_ticks, bpm, bpm_timeline=None):
    """tick → 秒。提供bpm_timeline时积分计算(变速谱正确), 否则用恒定BPM估算"""
    if bpm_timeline is None or len(bpm_timeline) <= 1:
        return (time_ticks / bpm) * 1.875  # ticks * 60 / (bpm * 32)
    # 积分: 从beat=0到time_ticks/32, 逐段累加时间
    target_beat = time_ticks / 32.0
    total_sec = 0.0
    for i, (seg_start, seg_bpm) in enumerate(bpm_timeline):
        seg_end = bpm_timeline[i+1][0] if i+1 < len(bpm_timeline) else target_beat
        seg_beats = max(0, min(seg_end, target_beat) - max(seg_start, 0))
        total_sec += seg_beats / seg_bpm * 60
    return total_sec


def collect_speed_events(judge_lines):
    all_events = []
    for line_idx, line in enumerate(judge_lines):
        for ev in line.get('speedEvents', []):
            all_events.append({
                'line_idx': line_idx,
                'startTime': ev.get('startTime', 0),
                'endTime': ev.get('endTime', 0),
                'value': ev.get('value', 1.0),
            })
    return all_events


def extract_features(chart_data, speed=1.0):
    """speed: 倍速缩放因子，用于调整所有秒级阈值（1/speed倍）"""
    all_notes, judge_lines, bpm_timeline = collect_all_notes(chart_data)
    if not all_notes:
        return None

    fallback_bpm = judge_lines[0].get('bpm', 120.0) if judge_lines else 120.0
    bpm = bpm_timeline[0][1] if bpm_timeline else fallback_bpm
    speed_events = collect_speed_events(judge_lines)

    n_notes = len(all_notes)
    times = np.array([n['time'] for n in all_notes])
    types = np.array([n['type'] for n in all_notes])
    positions = np.array([n.get('positionX', 0) for n in all_notes])
    hold_times = np.array([n.get('holdTime', 0) for n in all_notes])
    note_bpms = np.array([n['bpm'] for n in all_notes])  # 每个音符的真实BPM

    dt = float(times[-1]) if n_notes > 0 else 0
    features = {}

    features['total_notes'] = n_notes
    features['bpm'] = bpm
    features['judge_line_count'] = len(judge_lines)
    features['duration_beats'] = dt / 32  # ticks → beats
    features['duration_sec'] = _compute_duration_sec(bpm_timeline, features['duration_beats'])

    # BPM变化特征（从 BPMList 时间线获取）
    tl_bpms = [b for _, b in bpm_timeline] if bpm_timeline else [fallback_bpm]
    features['bpm_min'] = float(min(tl_bpms)) if tl_bpms else bpm
    features['bpm_max'] = float(max(tl_bpms)) if tl_bpms else bpm
    features['bpm_range'] = features['bpm_max'] - features['bpm_min']
    features['bpm_std'] = float(np.std(tl_bpms)) if len(tl_bpms) > 1 else 0.0
    features['bpm_change_count'] = len(tl_bpms) - 1

    # note type counts
    tap_mask = types == NOTE_TAP
    drag_mask = types == NOTE_DRAG
    hold_mask = types == NOTE_HOLD
    flick_mask = types == NOTE_FLICK

    n_tap = int(np.sum(tap_mask))
    n_drag = int(np.sum(drag_mask))
    n_hold = int(np.sum(hold_mask))
    n_flick = int(np.sum(flick_mask))

    features['tap_count'] = n_tap
    features['drag_count'] = n_drag
    features['hold_count'] = n_hold
    features['flick_count'] = n_flick
    features['tap_ratio'] = n_tap / max(n_notes, 1)
    features['drag_ratio'] = n_drag / max(n_notes, 1)
    features['hold_ratio'] = n_hold / max(n_notes, 1)
    features['flick_ratio'] = n_flick / max(n_notes, 1)
    features['drag_flick_ratio'] = (n_drag + n_flick) / max(n_notes, 1)  # 滑/粉占比（Phigros特有）

    ds = max(features['duration_sec'], 0.01)
    features['notes_per_second'] = n_notes / ds
    features['notes_per_beat'] = n_notes / max(dt, 0.01)
    features['tap_per_second'] = n_tap / ds
    features['tap_per_beat'] = n_tap / max(dt, 0.01)
    core_n = n_tap + n_hold  # 蓝键+长条 = 核心音符
    features['core_notes_per_second'] = core_n / ds
    features['core_notes_per_beat'] = core_n / max(dt, 0.01)

    # ====== 真实密度（排除 >1s 间隙，反映击打时真实密度水平） ======
    # 倍速时阈值等比缩放：2x速下0.5秒的间隙就相当于原速1秒
    rest_gap_threshold = 1.0 / speed
    all_t_sec = np.array([time_to_seconds(t, max(n.get('bpm', bpm), 1.0), bpm_timeline) for t, n in zip(times, all_notes)])
    all_t_sec.sort()
    if n_notes > 1:
        gaps = np.diff(all_t_sec)
        big_gaps = gaps[gaps > rest_gap_threshold]
        rest_duration = float(np.sum(big_gaps))
        real_active = max(all_t_sec[-1] - all_t_sec[0] - rest_duration, 0.01)
    else:
        rest_duration = 0.0
        real_active = max(ds, 0.01)
    features['real_active_sec'] = float(real_active)
    features['rest_duration_sec'] = float(rest_duration)
    features['rest_ratio'] = float(rest_duration / max(ds, 0.01))
    features['real_core_notes_per_second'] = core_n / real_active  # 真实核心TPS
    features['real_notes_per_second'] = n_notes / real_active  # 真实NPS

    # ====== 窗口密度（缓存避免重复计算） ======
    _density_cache = {}
    def _density(window):
        if window not in _density_cache:
            _density_cache[window] = _compute_window_density(times, window)
        return _density_cache[window]

    def _density_masked(mask, window):
        key = ('m', mask.sum(), window)
        if key not in _density_cache:
            _density_cache[key] = _compute_window_density(times[mask], window)
        return _density_cache[key]

    for w in [1, 2, 4, 8, 16]:
        d = _density(w)
        features[f'peak_density_{w}beat'] = float(np.max(d)) if d.size > 0 else 0
        features[f'mean_density_{w}beat'] = float(np.mean(d)) if d.size > 0 else 0
        features[f'std_density_{w}beat'] = float(np.std(d)) if d.size > 0 else 0
        p75 = float(np.percentile(d, 75)) if d.size > 0 else 0
        p90 = float(np.percentile(d, 90)) if d.size > 0 else 0
        features[f'p75_density_{w}beat'] = p75
        features[f'p90_density_{w}beat'] = p90
        total_d = float(np.sum(d))
        high_sum = float(np.sum(d[d >= p75])) if d.size > 0 and p75 > 0 else 0
        features[f'high_density_ratio_{w}beat'] = high_sum / max(total_d, 0.01)
        features[f'high_density_duration_{w}beat'] = float(_compute_high_duration(d, p75))
        features[f'high_density_duration_ratio_{w}beat'] = float(_compute_high_duration(d, p75) / max(len(d), 1))

    features['density_skew'] = float(np.mean(_density(4))) if _density(4).size > 0 else 0

    t4 = _density_masked(tap_mask, 4)
    features['peak_tap_density_4beat'] = float(np.max(t4)) if t4.size > 0 else 0
    features['mean_tap_density_4beat'] = float(np.mean(t4)) if t4.size > 0 else 0

    # ====== 多押（仅 Tap + Hold） ======
    core_mask = tap_mask | hold_mask
    core_notes = [all_notes[i] for i in range(n_notes) if core_mask[i]]
    core_times = times[core_mask]

    # core (tap+hold) 版本窗口密度（替代全音符版本）
    for w in [1, 2, 4, 8, 16]:
        cd = _compute_window_density(core_times, w)
        features[f'core_std_density_{w}beat'] = float(np.std(cd)) if cd.size > 0 else 0

    simultaneous = _compute_simultaneous_notes(core_notes)
    features['max_simultaneous'] = simultaneous['max']
    features['avg_simultaneous'] = simultaneous['avg']
    features['simultaneous_event_count'] = simultaneous['event_count']
    features['simultaneous_ratio'] = simultaneous['event_count'] / max(n_notes, 1)

    mf = simultaneous['multi_finger_events']
    features['multi_finger_3plus_events'] = mf['count_3plus']
    features['multi_finger_4plus_events'] = mf['count_4plus']
    features['multi_finger_3plus_ratio'] = mf['count_3plus'] / max(simultaneous['event_count'], 1)
    features['multi_finger_max_simultaneous'] = mf['max_simultaneous']

    # 加权多指协调分（区分流式/离散）
    # normalized: 总得分/时长，越高 = 多指协调越难
    features['weighted_mf_score_total'] = simultaneous['weighted_mf_score_total']
    features['weighted_mf_score_per_sec'] = simultaneous['weighted_mf_score_total'] / ds
    features['weighted_mf_score_mean'] = simultaneous['weighted_mf_score_mean']
    # 离散型多指占比：norm_spread > 1.5 的事件比例，越高越离散
    features['discrete_mf_ratio'] = simultaneous['discrete_mf_ratio']

    cs = simultaneous['chord_sizes']
    total_sim_ev = simultaneous['event_count']
    features['chord_2note_ratio'] = cs.get(2, 0) / max(total_sim_ev, 1)
    features['chord_3note_ratio'] = cs.get(3, 0) / max(total_sim_ev, 1)
    features['chord_4plus_ratio'] = (cs.get(4, 0) + cs.get(5, 0)) / max(total_sim_ev, 1)

    # 和弦大小熵（和弦大小分布的复杂度）
    if total_sim_ev > 0:
        cs_probs = np.array([cs.get(k, 0) for k in [2, 3, 4, 5]]) / max(total_sim_ev, 1)
        cs_probs = cs_probs[cs_probs > 0]
        features['chord_size_entropy'] = float(-np.sum(cs_probs * np.log2(cs_probs + 1e-10)))
    else:
        features['chord_size_entropy'] = 0.0

    ps = simultaneous['pos_spreads']
    features['sim_pos_spread_mean'] = float(np.mean(ps)) if ps else 0
    features['sim_pos_spread_max'] = float(np.max(ps)) if ps else 0

    features['multi_finger_density'] = total_sim_ev / ds

    mf_bursts = _compute_multifinger_bursts(core_notes)
    features['mf_burst_count'] = mf_bursts['count']
    features['mf_burst_avg_notes'] = mf_bursts['avg_notes']
    features['mf_burst_max_notes'] = mf_bursts['max_notes']
    features['mf_burst_avg_len_beats'] = mf_bursts['avg_len']
    features['mf_burst_max_len_beats'] = mf_bursts['max_len']

    # ====== 进阶多指（仅 Tap + Hold） ======
    _threshold = 0.03125
    core_windows = defaultdict(list)
    for n in core_notes:
        tk = round(n['time'] / _threshold) * _threshold
        core_windows[tk].append(n)

    mf_total = 0
    mf_with_hold = 0
    mf_cross_hand = 0
    for tk, notes in core_windows.items():
        if len(notes) >= 3:
            mf_total += 1
            if any(n['type'] == NOTE_HOLD for n in notes):
                mf_with_hold += 1
            pos = [n.get('positionX', 0) for n in notes]
            has_left = any(p < -0.3 for p in pos)
            has_right = any(p > 0.3 for p in pos)
            if has_left and has_right:
                mf_cross_hand += 1
    features['mf_with_hold_count'] = mf_with_hold
    features['mf_with_hold_ratio'] = mf_with_hold / max(mf_total, 1)

    mf_times = sorted([t for t, notes in core_windows.items() if len(notes) >= 3])
    features['dense_mf_count'] = sum(1 for i in range(1, len(mf_times)) if mf_times[i] - mf_times[i-1] <= 0.25)
    features['dense_mf_ratio'] = features['dense_mf_count'] / max(len(mf_times), 1) if len(mf_times) > 1 else 0

    features['mf_events_per_second'] = mf_total / ds

    # cross hand (all note types)
    all_windows = defaultdict(list)
    for n in all_notes:
        tk = round(n['time'] / _threshold) * _threshold
        all_windows[tk].append(n)
    cross_hand = 0
    for tk, notes in all_windows.items():
        if len(notes) >= 2:
            pos = [n.get('positionX', 0) for n in notes]
            if any(p < -0.3 for p in pos) and any(p > 0.3 for p in pos):
                cross_hand += 1
    features['cross_hand_event_count'] = cross_hand
    features['cross_hand_ratio'] = cross_hand / max(simultaneous['event_count'], 1)

    # ====== 进阶多指增强（跨线 = 真正的Phigros多指） ======
    mf_window = 0.03125
    multi_line_events = 0
    cross_line_3plus = 0
    for tk, notes in core_windows.items():
        if len(notes) >= 2:
            lines = set(n.get('judge_line_idx', 0) for n in notes)
            if len(lines) >= 2:
                multi_line_events += 1
            if len(notes) >= 3 and len(lines) >= 2:
                cross_line_3plus += 1
    features['multi_line_sim_events'] = multi_line_events
    features['multi_line_sim_ratio'] = multi_line_events / max(simultaneous['event_count'], 1)
    features['cross_line_3plus_count'] = cross_line_3plus

    total_chord = sum(simultaneous.get('chord_sizes', {}).values())
    total_chord_notes = sum(k * v for k, v in simultaneous.get('chord_sizes', {}).items())
    features['avg_chord_size'] = total_chord_notes / max(total_chord, 1)

    # ====== 锁手特征（向量化加速） ======
    if n_hold > 0:
        hold_t = times[hold_mask]
        hold_pos = positions[hold_mask]
        hold_len = hold_times[hold_mask]
        hold_end = hold_t + hold_len
        tap_t = times[tap_mask]
        tap_pos = positions[tap_mask]

        # 对每个hold，用二分查找找范围内的tap
        all_lock_events = 0
        total_disp = 0.0
        max_disp = 0.0
        tap_sorted_idx = np.argsort(tap_t)
        tap_t_sorted = tap_t[tap_sorted_idx]
        tap_pos_sorted = tap_pos[tap_sorted_idx]

        for hi in range(n_hold):
            left = np.searchsorted(tap_t_sorted, hold_t[hi], side='left')
            right = np.searchsorted(tap_t_sorted, hold_end[hi], side='right')
            if right > left:
                count = right - left
                all_lock_events += count
                disps = np.abs(tap_pos_sorted[left:right] - hold_pos[hi])
                total_disp += float(np.sum(disps))
                max_disp = max(max_disp, float(np.max(disps)))

        features['hold_lock_tap_events'] = all_lock_events
        features['hold_lock_tap_events_per_hold'] = all_lock_events / max(n_hold, 1)
        features['hold_lock_avg_displacement'] = total_disp / max(all_lock_events, 1)
        features['hold_lock_max_displacement'] = max_disp
        features['hold_lock_displacement_per_sec'] = total_disp / ds
    else:
        features.update({'hold_lock_tap_events': 0, 'hold_lock_tap_events_per_hold': 0,
                         'hold_lock_avg_displacement': 0, 'hold_lock_max_displacement': 0,
                         'hold_lock_displacement_per_sec': 0})

    # ====== 微窗口爆发（一次性算完） ======
    for mw in [0.0625, 0.125, 0.25]:
        d = _density(mw)
        if d.size > 0:
            sd = np.sort(d)[::-1]
            top5_n = max(5, len(d) // 20)
            features[f'micro_peak_top5_{mw}beat'] = float(np.mean(sd[:top5_n]))
            features[f'micro_spike_ratio_{mw}beat'] = float(sd[0] / max(np.mean(d), 0.01))
            features[f'micro_max_{mw}beat'] = float(sd[0])

    if core_times.size > 0:
        for mw in [0.0625, 0.125, 0.25]:
            d = _compute_window_density(core_times, mw)
            if d.size > 0:
                sd = np.sort(d)[::-1]
                top5_n = max(5, len(d) // 20)
                features[f'core_micro_max_{mw}beat'] = float(sd[0])
                features[f'core_micro_top5_{mw}beat'] = float(np.mean(sd[:top5_n]))

    # ====== Tap-only微窗口爆发（不用Drag/Flick充数） ======
    for mw in [0.0625, 0.125, 0.25]:
        d_tap = _density_masked(tap_mask, mw)
        if d_tap.size > 0:
            sd = np.sort(d_tap)[::-1]
            features[f'tap_micro_max_{mw}beat'] = float(sd[0])
            features[f'tap_micro_top5_{mw}beat'] = float(np.mean(sd[:max(5, len(d_tap)//20)]))

    # ====== 红蓝黄交替频率（Tap/Flick/Drag不同种连续切换的认知负荷） ======
    if n_notes > 2:
        nts = types.astype(int)
        switches = 0
        for i in range(1, n_notes):
            if times[i] - times[i-1] < 0.5 and nts[i] != nts[i-1]:
                switches += 1
        features['type_switch_ratio'] = switches / max(n_notes - 1, 1)
        features['type_switch_per_sec'] = switches / ds
    else:
        features.update({'type_switch_ratio': 0, 'type_switch_per_sec': 0})

    # ====== Tap纯密度（排除Drag/Flick） ======
    features['tap_notes_per_second'] = n_tap / ds
    features['tap_notes_per_beat'] = n_tap / max(dt, 0.01)

    # ====== 耐力 ======
    d1 = _density(1)
    if d1.size > 4:
        p75_d1 = float(np.percentile(d1, 75))
        runs = 0
        cur = 0
        for v in d1:
            if v >= p75_d1:
                cur += 1
            else:
                if cur >= 4:
                    runs += 1
                cur = 0
        if cur >= 4:
            runs += 1
        features['sustained_density_run_count'] = runs
        features['sustained_density_run_ratio'] = runs / max(len(d1) / 4, 1)

    # ====== 手速指数 ======
    tap_d1 = _density_masked(tap_mask, 1)
    if tap_d1.size > 2:
        td_mean = float(np.mean(tap_d1))
        td_max = float(np.max(tap_d1))
        td_sorted = np.sort(tap_d1)[::-1]
        top5_n = max(5, len(tap_d1) // 20)
        features['tap_burst_peak_to_mean'] = td_max / max(td_mean, 0.01)
        features['tap_burst_top5'] = float(np.mean(td_sorted[:top5_n]))
        p95 = float(np.percentile(tap_d1, 95))
        features['extreme_tap_window_ratio'] = float(np.sum(tap_d1 >= p95) / max(len(tap_d1), 1))

    tap_d05 = _density_masked(tap_mask, 0.5)
    if tap_d05.size > 0:
        sd05 = np.sort(tap_d05)[::-1]
        top5_n = max(5, len(tap_d05) // 20)
        features['tap_burst_05_top5'] = float(np.mean(sd05[:top5_n]))
        features['tap_burst_05_max'] = float(sd05[0])

    features['hand_speed_index'] = features.get('tap_per_second', 0) * features.get('tap_burst_peak_to_mean', 1)

    # ====== 定轨 ======
    if core_times.size > 0:
        beat_segments = defaultdict(list)
        for i in range(n_notes):
            if core_mask[i]:
                beat_segments[round(times[i])].append(positions[i])
        track_sections = 0
        for beat, pos_list in beat_segments.items():
            if len(pos_list) >= 3:
                rounded = [round(p * 2) / 2 for p in pos_list]
                unique_p = len(set(rounded))
                if unique_p <= 6:
                    from collections import Counter
                    mc = Counter(rounded).most_common(1)
                    if mc and mc[0][1] >= len(pos_list) * 0.6:
                        track_sections += 1
        features['track_section_count'] = track_sections
        features['track_section_ratio'] = track_sections / max(len(beat_segments), 1)
    else:
        features['track_section_count'] = 0
        features['track_section_ratio'] = 0

    # ====== 长条特征 ======
    hold_time_sum = float(np.sum(hold_times))
    features['total_hold_duration_beats'] = hold_time_sum
    features['total_hold_duration_sec'] = time_to_seconds(hold_time_sum, bpm)
    features['avg_hold_duration_beats'] = hold_time_sum / max(n_hold, 1)
    features['max_hold_duration_beats'] = float(np.max(hold_times)) if n_hold > 0 else 0
    features['hold_duration_ratio'] = hold_time_sum / max(dt, 0.01)

    # 同时长条
    concurrent_holds = _compute_concurrent_holds(all_notes)
    features['max_concurrent_holds'] = concurrent_holds['max']
    features['avg_concurrent_holds'] = concurrent_holds['avg']
    features['concurrent_hold_events'] = concurrent_holds['event_count']

    # ====== 位置特征 ======
    features['position_mean'] = float(np.mean(positions))
    features['position_std'] = float(np.std(positions))
    features['position_range'] = float(np.max(positions) - np.min(positions))
    features['position_abs_mean'] = float(np.mean(np.abs(positions)))
    features['position_iqr'] = float(np.percentile(positions, 75) - np.percentile(positions, 25))

    # ====== 节奏特征 ======
    if n_notes > 1:
        intervals = np.diff(times)
        features['avg_interval_beats'] = float(np.mean(intervals))
        features['std_interval_beats'] = float(np.std(intervals))
        features['min_interval_beats'] = float(np.min(intervals))
        features['interval_cv'] = float(np.std(intervals) / max(np.mean(intervals), 0.001))
        features['short_interval_ratio'] = float(np.sum(intervals < 0.25) / max(len(intervals), 1))
        features['very_short_interval_ratio'] = float(np.sum(intervals < 0.125) / max(len(intervals), 1))
    else:
        features.update({'avg_interval_beats': 0, 'std_interval_beats': 0, 'min_interval_beats': 0,
                         'interval_cv': 0, 'short_interval_ratio': 0, 'very_short_interval_ratio': 0})

    # ====== 位移 ======
    active_mask = tap_mask | flick_mask
    active_t = times[active_mask]
    active_pos = positions[active_mask]
    if len(active_t) > 1:
        gaps = np.abs(np.diff(active_t))
        pos_diffs = np.abs(np.diff(active_pos))
        valid = gaps <= 4
        distances = pos_diffs[valid]
        features['avg_movement'] = float(np.mean(distances)) if len(distances) > 0 else 0
        features['total_movement'] = float(np.sum(distances)) if len(distances) > 0 else 0
        features['max_movement'] = float(np.max(distances)) if len(distances) > 0 else 0
        features['movement_per_second'] = features['total_movement'] / ds
    else:
        features.update({'avg_movement': 0, 'total_movement': 0, 'max_movement': 0, 'movement_per_second': 0})

    # ====== speed events ======
    features['speed_event_count'] = len(speed_events)
    features['speed_event_density'] = len(speed_events) / max(ds, 0.01)  # 变速事件密度（次/秒）
    if speed_events:
        sv = np.array([ev['value'] for ev in speed_events])
        features['speed_mean'] = float(np.mean(sv))
        features['speed_std'] = float(np.std(sv))
        features['speed_max'] = float(np.max(sv))
        features['speed_min'] = float(np.min(sv))
        features['speed_range'] = float(np.ptp(sv))
    else:
        features.update({'speed_mean': 1.0, 'speed_std': 0, 'speed_max': 1.0, 'speed_min': 1.0, 'speed_range': 0})

    features['first_note_time'] = float(times[0])
    features['last_note_time'] = float(times[-1])

    na = sum(len(line.get('notesAbove', [])) for line in judge_lines)
    nb = sum(len(line.get('notesBelow', [])) for line in judge_lines)
    total_nb = na + nb
    features['notes_above_ratio'] = na / max(total_nb, 1)
    features['notes_below_ratio'] = nb / max(total_nb, 1)

    # ====== 节奏熵 ======
    if n_notes > 4:
        diffs = np.diff(np.sort(times))
        unique_v, counts = np.unique(np.round(diffs, 2), return_counts=True)
        probs = counts / max(np.sum(counts), 1)
        features['rhythm_entropy'] = float(-np.sum(probs * np.log2(probs + 1e-10)))
    else:
        features['rhythm_entropy'] = 0

    features['has_AT'] = 1 if n_flick > 0 else 0

    if n_notes > 4:
        times_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_timeline) for t, b in zip(times, note_bpms)])
        gaps_sec = np.diff(times_sec)
        features['max_gap_sec'] = float(np.max(gaps_sec))

    d4 = _density(4)
    if d4.size > 0:
        features['density_above_zero_ratio'] = float(np.sum(d4 > 0) / max(len(d4), 1))

    # core (tap+hold) 版本
    cd4 = _compute_window_density(core_times, 4) if core_times.size > 0 else np.array([])
    features['core_density_above_zero_ratio'] = float(np.sum(cd4 > 0) / max(len(cd4), 1)) if cd4.size > 0 else 0

    features['hold_interference_index'] = _compute_hold_interference_fast(all_notes, times, positions, hold_mask, n_hold, dt)

    # ====== 小窗口密度统计 ======
    for w in [0.25, 0.5, 1]:
        d = _density(w)
        if d.size > 0:
            sd = np.sort(d)[::-1]
            top5_n = max(5, len(d) // 20)
            features[f'peak_density_top5avg_{w}beat'] = float(np.mean(sd[:top5_n]))
            features[f'density_spike_ratio_{w}beat'] = float(sd[0] / max(np.mean(d), 0.01))
            features[f'peak_density_{w}beat'] = float(sd[0])
        # core (tap+hold) 版本
        cd = _compute_window_density(core_times, w) if core_times.size > 0 else np.array([])
        if cd.size > 0:
            csd = np.sort(cd)[::-1]
            ctop5_n = max(5, len(cd) // 20)
            features[f'core_peak_density_top5avg_{w}beat'] = float(np.mean(csd[:ctop5_n]))

    # ====== burst ======
    if n_notes > 4:
        d_half = _density(0.5)
        if d_half.size > 0:
            thresh = float(np.percentile(d_half, 90))
            burst_mask = d_half >= thresh
            features['burst_window_count'] = int(np.sum(burst_mask))
            features['burst_window_ratio'] = float(np.sum(burst_mask) / max(len(d_half), 1))
            # max consecutive burst
            runs = np.diff(np.concatenate(([False], burst_mask, [False])).astype(int))
            run_lengths = np.where(runs == -1)[0] - np.where(runs == 1)[0]
            features['max_consecutive_burst'] = int(np.max(run_lengths)) if len(run_lengths) > 0 else 0
            features['burst_intensity_mean'] = float(np.mean(d_half[burst_mask])) if np.any(burst_mask) else 0

    # ====== 滑动窗口峰值（1秒/0.5秒真实时间窗口，双指针法，tap+hold = core TPS） ======
    tps_mask = tap_mask | hold_mask
    tps_t = times[tps_mask]
    if len(tps_t) > 5:
        tps_bpm_arr = np.array([n.get('bpm', bpm) for n in all_notes])[tps_mask]
        tps_t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_timeline) for t, b in zip(tps_t, tps_bpm_arr)])
        tps_t_sec.sort()
        for win_name, win_sec in [('tps_1sec', 1.0), ('tps_05sec', 0.5)]:
            left = 0
            max_cnt = 0
            for right in range(len(tps_t_sec)):
                while tps_t_sec[right] - tps_t_sec[left] > win_sec:
                    left += 1
                max_cnt = max(max_cnt, right - left + 1)
            features[f'peak_{win_name}'] = int(max_cnt)
            if len(tps_t_sec) > 20:
                counts = []
                l = 0
                for r in range(len(tps_t_sec)):
                    while tps_t_sec[r] - tps_t_sec[l] > win_sec:
                        l += 1
                    counts.append(r - l + 1)
                counts.sort(reverse=True)
                top5 = min(5, len(counts) // 20) if len(counts) > 20 else min(5, len(counts))
                features[f'peak_{win_name}_top5avg'] = float(np.mean(counts[:max(top5, 1)]))
    else:
        for win_name in ['tps_1sec', 'tps_05sec']:
            features[f'peak_{win_name}'] = 0
            features[f'peak_{win_name}_top5avg'] = 0.0

    # ====== 1秒窗口全音符密度峰值（1smax密度） ======
    all_t_sec = np.array([time_to_seconds(t, max(n.get('bpm', bpm), 1.0), bpm_timeline) for t, n in zip(times, all_notes)])
    all_t_sec.sort()
    if len(all_t_sec) > 5:
        left = 0; max_cnt = 0; all_counts = []
        for right in range(len(all_t_sec)):
            while all_t_sec[right] - all_t_sec[left] > 1.0:
                left += 1
            max_cnt = max(max_cnt, right - left + 1)
            all_counts.append(right - left + 1)
        features['peak_density_1sec'] = int(max_cnt)
        all_counts.sort(reverse=True)
        features['peak_density_1sec_top5avg'] = float(np.mean(all_counts[:max(5, len(all_counts)//20)]))
    else:
        features['peak_density_1sec'] = 0
        features['peak_density_1sec_top5avg'] = 0.0

    # ====== 1秒窗口核心音符密度峰值（tap+hold，不含drag/flick） ======
    core_mask_1s = tap_mask | hold_mask
    core_notes_times = times[core_mask_1s]
    if len(core_notes_times) > 5:
        core_t_sec_1s = np.array([time_to_seconds(t, max(all_notes[i].get('bpm', bpm), 1.0), bpm_timeline)
                                   for i, t in enumerate(core_notes_times)])
        core_t_sec_1s.sort()
        left = 0; max_cnt = 0; all_counts = []
        for right in range(len(core_t_sec_1s)):
            while core_t_sec_1s[right] - core_t_sec_1s[left] > 1.0:
                left += 1
            max_cnt = max(max_cnt, right - left + 1)
            all_counts.append(right - left + 1)
        features['core_peak_density_1sec'] = int(max_cnt)
        all_counts.sort(reverse=True)
        features['core_peak_density_1sec_top5avg'] = float(np.mean(all_counts[:max(5, len(all_counts)//20)]))
    else:
        features['core_peak_density_1sec'] = 0
        features['core_peak_density_1sec_top5avg'] = 0.0

    # ====== 密度维度（保留旧公式，above_avg 在前端独立展示） ======
    rcnps = features.get('real_core_notes_per_second', 0)
    cp1s = features.get('core_peak_density_1sec_top5avg', 0)
    features['density_dimension'] = float(np.sqrt(max(rcnps, 0.01) * max(cp1s, 0.01)))

    # ====== 高于均值的1s峰值TPS（前端展示用，反映高潮段密度） ======
    above_avg_density = cp1s  # fallback
    if len(core_t_sec_1s_all := np.array([time_to_seconds(t, max(all_notes[i].get('bpm', bpm), 1.0), bpm_timeline)
                                           for i, t in enumerate(core_times)])) > 5:
        t_arr = np.sort(core_t_sec_1s_all)
        left = 0; above_counts = []
        for right in range(len(t_arr)):
            while t_arr[right] - t_arr[left] > 1.0:
                left += 1
            cnt = right - left + 1
            if cnt >= rcnps:
                above_counts.append(cnt)
        if above_counts:
            above_counts.sort(reverse=True)
            top5 = above_counts[:max(5, len(above_counts)//20)]
            above_avg_density = float(np.mean(top5))
    features['above_avg_density_1sec_top5'] = above_avg_density

    # ====== 耐力指标：1秒窗口tps>平均*0.9 的秒数占比 ======
    stamina_mask = tap_mask | hold_mask  # 蓝键+长条 = core notes
    core_t = times[stamina_mask]
    if len(core_t) > 10:
        core_bpm_arr = np.array([n.get('bpm', bpm) for n in all_notes])[stamina_mask]
        core_t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_timeline) for t, b in zip(core_t, core_bpm_arr)])
        core_t_sec.sort()
        # stamina ratio uses total duration (ds) for avg TPS, per original code
        avg_core_tps = len(core_t_sec) / max(ds, 0.01)
        threshold = avg_core_tps * 0.9
        # Sliding window count > threshold
        left = 0; high_sec = 0; total_windows = 0
        prev_t = None
        for right in range(len(core_t_sec)):
            while core_t_sec[right] - core_t_sec[left] > 1.0:
                left += 1
            cnt = right - left + 1
            total_windows += 1
            if cnt >= threshold:
                high_sec += 1
        features['stamina_high_sec'] = high_sec
        features['stamina_ratio'] = high_sec / max(total_windows, 1)
    else:
        features['stamina_high_sec'] = 0
        features['stamina_ratio'] = 0.0

    # ====== jack (按同线同位置分组, 使用真实秒数阈值) ======
    if n_notes > 3:
        intervals = np.diff(times)
        bpm_arr = np.array([n.get('bpm', 120.0) for n in all_notes])
        intervals_sec = np.array([
            time_to_seconds(intervals[i], max(bpm_arr[i], 1.0))
            for i in range(len(intervals))
        ])

        # global: 极短间隔密度指标 (保持原有微窗口密度测量)
        features['global_jack_count'] = int(np.sum(intervals < 0.125))
        features['miniburst_count'] = int(np.sum(intervals < 0.0625))
        features['miniburst_density'] = features['miniburst_count'] / max(dt, 0.01)

        # position-aware jack: 同线同位置且间隔 < 100ms
        jack_threshold_sec = 0.10
        jl_idx = np.array([n.get('judge_line_idx', 0) for n in all_notes])
        pos_rounded = np.round(positions, 1)

        same_pos = np.zeros(n_notes, dtype=bool)
        for i in range(1, n_notes):
            if (jl_idx[i] == jl_idx[i-1] and pos_rounded[i] == pos_rounded[i-1]
                    and intervals_sec[i-1] < jack_threshold_sec):
                same_pos[i] = True
                same_pos[i-1] = True

        features['same_line_jack_count'] = int(np.sum(same_pos))
        features['same_line_jack_ratio'] = features['same_line_jack_count'] / max(len(intervals), 1)

        runs = np.diff(np.concatenate(([False], same_pos, [False])).astype(int))
        run_starts = np.where(runs == 1)[0]
        run_ends = np.where(runs == -1)[0]
        run_lengths = run_ends - run_starts
        short_jack_runs = [l for l in run_lengths if 2 <= l <= 3]
        long_jack_runs = [l for l in run_lengths if l >= 4]
        features['short_jack_count'] = int(np.sum(short_jack_runs))
        features['long_jack_count'] = int(np.sum(long_jack_runs))
        features['jack_max_run'] = int(np.max(run_lengths)) if len(run_lengths) > 0 else 0

        tempo_change = 0
        for i in range(1, len(intervals)):
            if intervals[i] > intervals[i-1] * 1.5 or intervals[i] < intervals[i-1] * 0.67:
                tempo_change += 1
        features['tempo_change_count'] = tempo_change
        features['tempo_change_ratio'] = tempo_change / max(len(intervals), 1)

    # ====== 楼梯/Scale模式 v2（和弦感知 + 秒归一化 + 速度加权） ======
    if n_notes > 3:
        duration_sec = features['duration_sec']  # 已用BPM时间线正确计算
        
        # Step 1: 按时间分组（5ms内=同一和弦事件），取中位数位置
        times_sec_arr = np.array([time_to_seconds(t, max(b, 1.0), bpm_timeline) for t, b in zip(times, note_bpms)])
        events = []  # [(time_sec, median_pos, chord_size)]
        i = 0
        while i < n_notes:
            t0 = times_sec_arr[i]
            group_positions = [positions[i]]
            j = i + 1
            while j < n_notes and (times_sec_arr[j] - t0) < 0.005:
                group_positions.append(positions[j])
                j += 1
            events.append((t0, float(np.median(group_positions)), len(group_positions)))
            i = j
        
        n_events = len(events)
        if n_events >= 4:
            event_times = np.array([e[0] for e in events])
            event_positions = np.array([e[1] for e in events])
            event_chord_sizes = np.array([e[2] for e in events])
            
            # 事件间间隔（秒）
            event_intervals = np.diff(event_times)
            
            # Step 2: 检测楼梯段（连续同方向位移，间隔<0.15s）
            stair_threshold = 0.15  # 最大间隔仍算同一楼梯段
            stair_runs = []  # [(start_idx, end_idx, total_steps, direction_changes)]
            
            ei = 1
            while ei < n_events:
                if event_intervals[ei-1] < stair_threshold:
                    dirs = []
                    j = ei
                    while j < n_events and event_intervals[j-1] < stair_threshold:
                        dp = event_positions[j] - event_positions[j-1]
                        if abs(dp) > 0.05:
                            dirs.append(1 if dp > 0 else -1)
                        else:
                            dirs.append(0)
                        j += 1
                    
                    if len(dirs) >= 3:
                        # 统计方向变化
                        dir_changes = sum(1 for k in range(1, len(dirs)) if dirs[k] != dirs[k-1] and dirs[k] != 0 and dirs[k-1] != 0)
                        # 统计同方向连续步数（纯楼梯特征）
                        same_dir_runs = 0
                        cur_run = 0
                        for k in range(1, len(dirs)):
                            if dirs[k] == dirs[k-1] and dirs[k] != 0:
                                cur_run += 1
                            else:
                                if cur_run >= 1:
                                    same_dir_runs += cur_run + 1
                                cur_run = 0
                        if cur_run >= 1:
                            same_dir_runs += cur_run + 1
                        
                        total_steps = len(dirs)
                        stair_runs.append((ei, j, total_steps, dir_changes, same_dir_runs))
                    ei = j
                else:
                    ei += 1
            
            if stair_runs:
                # 汇总指标
                total_stair_steps = sum(r[2] for r in stair_runs)
                total_dir_changes = sum(r[3] for r in stair_runs)
                total_same_dir = sum(r[4] for r in stair_runs)
                
                # 楼梯速度（步/秒）
                features['stair_rate_per_sec'] = total_stair_steps / max(duration_sec, 0.01)
                
                # 楼梯复杂度：方向变化越多越难（山峰形 > 单边爬升）
                features['stair_complexity'] = total_dir_changes / max(total_stair_steps, 1)
                
                # 同方向步数占比（纯楼梯特征，越高=越像传统楼梯）
                features['stair_purity'] = total_same_dir / max(total_stair_steps, 1)
                
                # 楼梯速度加权：对高速楼梯段做额外加分
                stair_speeds = []
                for run in stair_runs:
                    seg_start = run[0] - 1
                    seg_end = min(run[1], n_events) - 1
                    seg_dur = event_times[seg_end] - event_times[seg_start]
                    if seg_dur > 0.001:
                        stair_speeds.append(run[2] / seg_dur)
                features['stair_speed_avg'] = float(np.mean(stair_speeds)) if stair_speeds else 0.0
                features['stair_speed_max'] = float(np.max(stair_speeds)) if stair_speeds else 0.0
                
                # 和弦楼梯：楼梯段中涉及和弦（chord_size>1）的占比
                chord_stair_events = 0
                for run in stair_runs:
                    for k in range(run[0]-1, min(run[1], n_events)):
                        if event_chord_sizes[k] >= 2:
                            chord_stair_events += 1
                features['stair_chord_ratio'] = chord_stair_events / max(total_stair_steps, 1)
                
                # 保留兼容旧名
                features['stair_total_steps'] = total_stair_steps
                features['stair_climb_count'] = len(stair_runs)
                features['stair_density'] = features['stair_rate_per_sec']  # 覆盖为秒归一化
            else:
                features.update({
                    'stair_rate_per_sec': 0, 'stair_complexity': 0, 'stair_purity': 0,
                    'stair_speed_avg': 0, 'stair_speed_max': 0, 'stair_chord_ratio': 0,
                    'stair_total_steps': 0, 'stair_climb_count': 0, 'stair_density': 0,
                })
        else:
            features.update({
                'stair_rate_per_sec': 0, 'stair_complexity': 0, 'stair_purity': 0,
                'stair_speed_avg': 0, 'stair_speed_max': 0, 'stair_chord_ratio': 0,
                'stair_total_steps': 0, 'stair_climb_count': 0, 'stair_density': 0,
            })
    else:
        features.update({
            'stair_rate_per_sec': 0, 'stair_complexity': 0, 'stair_purity': 0,
            'stair_speed_avg': 0, 'stair_speed_max': 0, 'stair_chord_ratio': 0,
            'stair_total_steps': 0, 'stair_climb_count': 0, 'stair_density': 0,
        })

    # ====== 颤音/Trill密度（连续左右交替，参考ManiaMapAnalyser的Trill/Minitrill） ======
    if n_notes > 3:
        trill_threshold_sec = 0.10
        trill_events = 0
        trill_total_steps = 0
        i = 1
        while i < n_notes:
            if intervals_sec[i-1] < trill_threshold_sec:
                dirs = []
                j = i
                while j < n_notes and intervals_sec[j-1] < trill_threshold_sec:
                    diff_pos = positions[j] - positions[j-1]
                    if abs(diff_pos) > 0.01:
                        dirs.append(1 if diff_pos > 0 else -1)
                    j += 1
                if len(dirs) >= 3:
                    alt_count = sum(1 for k in range(1, len(dirs)) if dirs[k] != dirs[k-1])
                    # 严格交替：每次方向都反转 = 纯颤音
                    if alt_count >= len(dirs) - 1:
                        trill_events += 1
                        trill_total_steps += len(dirs)
                i = j
            else:
                i += 1
        features['trill_event_count'] = trill_events
        features['trill_total_steps'] = trill_total_steps
        features['trill_density'] = trill_total_steps / max(dt, 0.01)
    else:
        features.update({'trill_event_count': 0, 'trill_total_steps': 0, 'trill_density': 0})

    # ====== 纵连密度（jack density — 同位置连续击打频率） ======
    if n_notes > 3:
        jack_events = 0
        jack_total_steps = 0
        i = 1
        while i < n_notes:
            if intervals_sec[i-1] < jack_threshold_sec:
                j = i
                same_pos_run = 0
                while j < n_notes and intervals_sec[j-1] < jack_threshold_sec:
                    if abs(positions[j] - positions[j-1]) < 0.01 and jl_idx[j] == jl_idx[j-1]:
                        same_pos_run += 1
                    else:
                        if same_pos_run >= 2:
                            jack_events += 1
                            jack_total_steps += same_pos_run + 1
                        same_pos_run = 0
                    j += 1
                if same_pos_run >= 2:
                    jack_events += 1
                    jack_total_steps += same_pos_run + 1
                i = j
            else:
                i += 1
        features['jack_event_count'] = jack_events
        features['jack_total_steps'] = jack_total_steps
        features['jack_density'] = jack_total_steps / max(dt, 0.01)
    else:
        features.update({'jack_event_count': 0, 'jack_total_steps': 0, 'jack_density': 0})

    # ====== 左右分布 ======
    act_mask = tap_mask | flick_mask
    act_pos = positions[act_mask]
    if np.sum(act_mask) > 3:
        n_act = int(np.sum(act_mask))
        left = np.sum(act_pos < -0.5)
        right = np.sum(act_pos > 0.5)
        center = np.sum(np.abs(act_pos) <= 0.5)
        features['left_ratio'] = left / max(n_act, 1)
        features['right_ratio'] = right / max(n_act, 1)
        features['center_ratio'] = center / max(n_act, 1)
        features['spread_balance'] = abs(left - right) / max(left + right, 1)

        act_t = times[act_mask]
        burst_moves = []
        for i in range(1, n_act):
            if act_t[i] - act_t[i-1] < 0.5:
                burst_moves.append(abs(act_pos[i] - act_pos[i-1]))
        features['burst_avg_movement'] = float(np.mean(burst_moves)) if burst_moves else 0
        features['burst_max_movement'] = float(np.max(burst_moves)) if burst_moves else 0
        features['burst_movement_ratio'] = len(burst_moves) / max(n_act, 1)
    else:
        features.update({'left_ratio': 0, 'right_ratio': 0, 'center_ratio': 0, 'spread_balance': 0,
                         'burst_avg_movement': 0, 'burst_max_movement': 0, 'burst_movement_ratio': 0})

    # ====== 节奏多样性 ======
    if n_notes > 2:
        diffs = np.diff(np.sort(times))
        unique_v, counts = np.unique(np.round(diffs, 3), return_counts=True)
        features['distinct_rhythm_count'] = len(unique_v)
        features['rhythm_diversity'] = len(unique_v) / max(dt, 0.01)
        features['dominant_rhythm_ratio'] = float(np.max(counts) / max(np.sum(counts), 1))
    else:
        features.update({'distinct_rhythm_count': 0, 'rhythm_diversity': 0, 'dominant_rhythm_ratio': 0})

    # ====== 读谱：clutter ======
    if n_notes > 1:
        clutter = 0
        for i in range(1, n_notes):
            if times[i] - times[i-1] < 0.04 and abs(positions[i] - positions[i-1]) > 0.5:
                clutter += 1
        features['note_clutter_count'] = clutter
        features['note_clutter_ratio'] = clutter / max(n_notes, 1)
    else:
        features.update({'note_clutter_count': 0, 'note_clutter_ratio': 0})

    # ====== offbeat ======
    offbeat = int(np.sum(np.abs(times - np.round(times)) > 0.05))
    features['offbeat_ratio'] = offbeat / max(n_notes, 1)
    weak = int(np.sum(np.abs((times + 0.5) % 1.0 - 0.5) < 0.05))
    features['weak_beat_ratio'] = weak / max(n_notes, 1)

    # ====== density transition ======
    if d4.size > 2:
        dc = np.abs(np.diff(d4))
        features['density_transition_mean'] = float(np.mean(dc))
        features['density_transition_max'] = float(np.max(dc))
        features['density_transition_std'] = float(np.std(dc))
    else:
        features.update({'density_transition_mean': 0, 'density_transition_max': 0, 'density_transition_std': 0})

    # ====== stop-go ======
    if d4.size > 4:
        p75_4 = float(np.percentile(d4, 75))
        p25_4 = float(np.percentile(d4, 25))
        sg = 0
        for i in range(1, len(d4)):
            if (d4[i-1] > p75_4 and d4[i] < p25_4) or (d4[i-1] < p25_4 and d4[i] > p75_4):
                sg += 1
        features['stop_go_count'] = sg
        features['stop_go_ratio'] = sg / max(len(d4), 1)
    else:
        features.update({'stop_go_count': 0, 'stop_go_ratio': 0})

    # ====== speed change ======
    if len(speed_events) > 1:
        sv = np.array([abs(ev['value'] - 1.0) for ev in speed_events])
        features['speed_change_total_impact'] = float(np.sum(sv))
        features['speed_change_max_impact'] = float(np.max(sv))
        features['speed_change_mean_impact'] = float(np.mean(sv))
    else:
        features.update({'speed_change_total_impact': 0, 'speed_change_max_impact': 0, 'speed_change_mean_impact': 0})

    # ====== hold-tap overlap ======
    if n_hold > 0:
        hold_start = times[hold_mask]
        hold_end = hold_start + hold_times[hold_mask]
        tap_flick_mask = tap_mask | flick_mask
        tap_f_t = times[tap_flick_mask]
        overlap = 0
        for hi in range(n_hold):
            left = np.searchsorted(tap_f_t, hold_start[hi], side='left')
            right = np.searchsorted(tap_f_t, hold_end[hi], side='right')
            if right > left:
                overlap += 1
        features['hold_tap_overlap_count'] = overlap
        features['hold_tap_overlap_ratio'] = overlap / max(n_hold, 1)
    else:
        features.update({'hold_tap_overlap_count': 0, 'hold_tap_overlap_ratio': 0})

    # ====== wide jumps ======
    if n_notes > 1:
        time_gaps = np.diff(times)
        pos_gaps = np.abs(np.diff(positions))
        wide = int(np.sum((time_gaps < 0.25) & (pos_gaps > 1.5)))
        features['wide_jump_count'] = wide
        features['wide_jump_density'] = wide / max(dt, 0.01)
    else:
        features.update({'wide_jump_count': 0, 'wide_jump_density': 0})

    features['visual_complexity'] = float(sum(simultaneous.get('chord_sizes', {}).values()) / max(simultaneous['event_count'], 1)) if simultaneous['event_count'] > 0 else 0

    # ====== position entropy ======
    if len(positions) > 5:
        hist, _ = np.histogram(positions, bins=10, range=(-2, 2))
        prob = hist / max(np.sum(hist), 1)
        features['position_entropy'] = float(-np.sum(prob * np.log2(prob + 1e-10)))
    else:
        features['position_entropy'] = 0

    # ====== 位置聚类 & 离轨度（反映谱面是"定轨"还是"散乱"） ======
    if len(positions) > 10:
        # 用K-means-like方法：以0.5为步长扫描，统计"虚拟轨道"数量
        # 轨道定义为：在该positionX附近聚集了足够多音符的区域
        bucket_size = 0.3
        pos_buckets = defaultdict(int)
        for p in positions:
            bucket = round(p / bucket_size) * bucket_size
            pos_buckets[bucket] += 1
        # 显著轨道：该bucket内音符数 >= 总音符的3%
        sig_threshold = max(n_notes * 0.03, 3)
        significant_tracks = sum(1 for cnt in pos_buckets.values() if cnt >= sig_threshold)
        features['position_cluster_count'] = significant_tracks

        # 离轨度：每个音符到最近"轨道中心"的距离的平均值
        if significant_tracks > 0:
            track_centers = sorted([k for k, cnt in pos_buckets.items() if cnt >= sig_threshold])
            if track_centers:
                deviations = []
                for p in positions:
                    min_dist = min(abs(p - tc) for tc in track_centers)
                    deviations.append(min_dist)
                features['track_deviation_score'] = float(np.mean(deviations))
            else:
                features['track_deviation_score'] = 0.0
        else:
            features['track_deviation_score'] = 0.0
    else:
        features['position_cluster_count'] = 0
        features['track_deviation_score'] = 0.0

    # ====== 单指 TPS (per-finger TPS, 参考osu!mania定轨分析) ======
    # 区分: 拍拍谱(多指和弦交替, 单指负载低) vs 交互谱(单点交替, 单指负载高)
    # 将位置离散化为6个"手指"通道, 滑动窗口统计各通道密度
    if n_notes > 10:
        tap_only = times[tap_mask]
        tap_pos = positions[tap_mask]
        if len(tap_only) > 5:
            bucket_edges = np.array([-4.5, -3.0, -1.5, 0, 1.5, 3.0, 4.5])
            finger_idx = np.clip(np.digitize(tap_pos, bucket_edges) - 1, 0, 5)
            window_sec = 0.5
            window_beats = window_sec * bpm / 1.875
            step_beats = max(window_beats / 4, 0.01)
            t0 = times[0]
            t_end = times[-1]
            cur = t0
            all_finger_peaks = []
            max_peak = 0
            while cur < t_end:
                win_end = cur + window_beats
                m = (tap_only >= cur) & (tap_only < win_end)
                if np.sum(m) > 1:
                    counts = np.bincount(finger_idx[m], minlength=6)
                    peak = np.max(counts) / window_sec
                    all_finger_peaks.append(peak)
                    max_peak = max(max_peak, peak)
                cur += step_beats
            features['finger_peak_tps'] = round(max_peak, 4)
            features['finger_avg_peak_tps'] = round(np.mean(all_finger_peaks), 4) if all_finger_peaks else 0
            overall_tps = features.get('tap_per_second', 1)
            features['finger_vs_total_ratio'] = round(max_peak / max(overall_tps, 0.01), 4)
        else:
            features.update({'finger_peak_tps': 0, 'finger_avg_peak_tps': 0, 'finger_vs_total_ratio': 0})
    else:
        features.update({'finger_peak_tps': 0, 'finger_avg_peak_tps': 0, 'finger_vs_total_ratio': 0})

    # ====== 型切换频率（pattern switch rate） ======
    # FIX: 用0.5秒滑动窗口替代1拍窗口，加权不同型之间的切换烈度
    if n_notes > 10:
        window_sec = 0.5
        seg_labels = []
        step_sec = 0.25  # 滑动步长
        cur_sec = 0.0
        ts_sec = times_sec.copy() if 'times_sec' in dir() else np.array([time_to_seconds(t, max(b, 1.0), bpm_timeline) for t, b in zip(times, note_bpms)])
        t_end_sec = float(ts_sec[-1])
        while cur_sec + window_sec <= t_end_sec:
            win_end_sec = cur_sec + window_sec
            m = (ts_sec >= cur_sec) & (ts_sec < win_end_sec)
            n_win = int(np.sum(m))
            if n_win <= 1:
                seg_labels.append('空')
            else:
                w_pos = positions[m]
                w_times_beat = times[m]
                chord_count = 0
                for k in range(1, n_win):
                    if w_times_beat[k] - w_times_beat[k-1] < 0.0625:  # 放宽到1/16拍
                        chord_count += 1
                if chord_count >= n_win * 0.25:
                    seg_labels.append('和弦')
                else:
                    dirs = []
                    jacks = 0
                    for k in range(1, n_win):
                        diff = w_pos[k] - w_pos[k-1]
                        if abs(diff) < 0.02:
                            jacks += 1
                        elif diff > 0:
                            dirs.append(1)
                        else:
                            dirs.append(-1)
                    if jacks >= n_win * 0.4:
                        seg_labels.append('纵连')
                    elif len(dirs) >= 3:
                        alt_count = sum(1 for k in range(1, len(dirs)) if dirs[k] != dirs[k-1])
                        if alt_count >= len(dirs) * 0.8:
                            seg_labels.append('颤音')
                        elif alt_count == 0:
                            seg_labels.append('楼梯')
                        else:
                            seg_labels.append('流谱')
                    elif len(dirs) >= 2:
                        if all(d == dirs[0] for d in dirs):
                            seg_labels.append('楼梯')
                        else:
                            seg_labels.append('流谱')
                    elif len(dirs) == 1:
                        seg_labels.append('流谱')
                    else:
                        seg_labels.append('单点')
            cur_sec += step_sec

        # 统计型切换次数并加权烈度
        if len(seg_labels) > 1:
            # 定义切换烈度权重：型差距越大权重越高
            severity = {
                ('空',): 0,
                ('单点', '空'): 0.5, ('流谱', '空'): 0.5, ('空', '流谱'): 0.5, ('空', '单点'): 0.5,
                ('流谱', '楼梯'): 0.7, ('楼梯', '流谱'): 0.7,
                ('流谱', '颤音'): 0.8, ('颤音', '流谱'): 0.8,
                ('流谱', '和弦'): 0.9, ('和弦', '流谱'): 0.9,
                ('楼梯', '颤音'): 0.9, ('颤音', '楼梯'): 0.9,
                ('流谱', '纵连'): 1.0, ('纵连', '流谱'): 1.0,
                ('楼梯', '和弦'): 1.0, ('和弦', '楼梯'): 1.0,
                ('颤音', '和弦'): 1.2, ('和弦', '颤音'): 1.2,
                ('纵连', '和弦'): 1.2, ('和弦', '纵连'): 1.2,
                ('颤音', '纵连'): 1.3, ('纵连', '颤音'): 1.3,
            }
            switches = 0
            weighted = 0.0
            for i in range(1, len(seg_labels)):
                a, b = seg_labels[i-1], seg_labels[i]
                if a == b:
                    continue
                if a == '空' and b == '空':
                    continue
                key = tuple(sorted([a, b]))
                w = severity.get(key, 0.8)
                if w > 0:
                    switches += 1
                    weighted += w
            features['pattern_switch_count'] = switches
            features['pattern_switch_rate'] = switches / max(len(seg_labels) * step_sec, 0.01)
        else:
            features['pattern_switch_count'] = 0
            features['pattern_switch_rate'] = 0.0
    else:
        features['pattern_switch_count'] = 0
        features['pattern_switch_rate'] = 0.0

    # ====== 和弦交替率（chord→single→chord frequency） ======
    if n_notes > 5:
        chord_alt_count = 0
        in_chord = False
        # 用0.01拍作为同位判定
        for i in range(1, n_notes):
            if times[i] - times[i-1] < 0.02:
                if not in_chord:
                    in_chord = True
            else:
                if in_chord:
                    chord_alt_count += 1
                    in_chord = False
        features['chord_alternation_rate'] = chord_alt_count / max(ds, 0.01)
    else:
        features['chord_alternation_rate'] = 0.0

    # ====== 方向不规则度（direction change entropy — 高阶方向变化的熵） ======
    if n_notes > 5:
        dirs = []
        pos_prev = float(positions[0])
        for i in range(1, n_notes):
            d = float(positions[i]) - pos_prev
            if abs(d) > 0.01:
                dirs.append(1 if d > 0 else -1)
            pos_prev = float(positions[i])
        if len(dirs) >= 4:
            # 二阶方向变化：看相邻一阶方向是相同还是相反
            second_order = []
            for i in range(1, len(dirs)):
                so = 1 if dirs[i] == dirs[i-1] else -1
                second_order.append(so)
            if second_order:
                from collections import Counter
                cnt = Counter(second_order)
                total = len(second_order)
                p_same = cnt.get(1, 0) / total
                p_diff = cnt.get(-1, 0) / total
                # 越接近p=0.5越不规则（纯随机=高熵），越接近p=1=平滑流（低熵）
                def safe_entropy(p):
                    if p <= 0 or p >= 1:
                        return 0
                    return -p * np.log2(p)
                features['direction_irregularity'] = float(safe_entropy(p_same) + safe_entropy(p_diff))
            else:
                features['direction_irregularity'] = 0.0
        else:
            features['direction_irregularity'] = 0.0
    else:
        features['direction_irregularity'] = 0.0

    # ====== 位置熵（position entropy — 音符在x轴分布的不均匀度） ======
    if n_notes > 3:
        # 将x轴分为20个bucket，计算分布熵
        x_buckets = np.linspace(-1.0, 1.0, 21)
        x_indices = np.digitize(positions, x_buckets) - 1
        x_indices = np.clip(x_indices, 0, 19)
        bucket_counts = np.bincount(x_indices, minlength=20)
        probs = bucket_counts / n_notes
        probs = probs[probs > 0]
        features['position_entropy'] = float(-np.sum(probs * np.log2(probs + 1e-10)))
        # 位置范围（实际使用的x范围，归一化）
        x_min, x_max = float(np.min(positions)), float(np.max(positions))
        x_span = max(abs(x_max), abs(x_min))
        features['position_range_used'] = (x_max - x_min) / max(x_span * 2.0, 0.1)
    else:
        features['position_entropy'] = 0.0
        features['position_range_used'] = 0.0

    # ====== v8.0 新配置特征：快音符密度 + 和弦size + 节奏种类（仅核心note: tap+hold）======
    core_mask_v8 = (types == NOTE_TAP) | (types == NOTE_HOLD)
    core_notes_v8 = [all_notes[i] for i in range(n_notes) if core_mask_v8[i]]
    dur = features.get('duration_sec', 1.0)
    if len(core_notes_v8) > 1 and dur > 0:
        # ① 同线间隔分析 (BPM归一化, 仅核心note)
        fast_16th = 0; rhythm_counts = {}
        for i, n0 in enumerate(core_notes_v8):
            line0 = n0.get('judge_line_idx', 0)
            t0_sec = time_to_seconds(n0['time'], max(n0.get('bpm', bpm), 1.0), bpm_timeline)
            for j in range(i + 1, min(i + 50, len(core_notes_v8))):
                nj = core_notes_v8[j]
                if nj.get('judge_line_idx', 0) != line0: continue
                tj_sec = time_to_seconds(nj['time'], max(nj.get('bpm', bpm), 1.0), bpm_timeline)
                gap_sec = tj_sec - t0_sec
                if gap_sec <= 0.005: continue
                avg_bpm_val = (n0.get('bpm', bpm) + nj.get('bpm', bpm)) / 2
                beats_val = gap_sec * avg_bpm_val / 60.0
                if beats_val > 1.5: break
                matched = None
                for frac, target in [(2,0.5),(3,1/3),(4,0.25),(5,0.2),(6,1/6),(7,1/7),
                                      (8,0.125),(9,1/9),(12,1/12),(14,1/14),(16,0.0625),
                                      (24,1/24),(28,1/28),(32,0.03125)]:
                    if abs(beats_val - target) / max(target, 0.001) < 0.12:
                        matched = frac; break
                if matched:
                    if matched >= 4: fast_16th += 1
                    if matched >= 2: rhythm_counts[matched] = rhythm_counts.get(matched, 0) + 1
                    break
                elif beats_val > 0.02:
                    rhythm_counts[0] = rhythm_counts.get(0, 0) + 1; break
        features['fast_note_density_16th'] = fast_16th / max(dur, 0.01)
        features['rhythm_type_count'] = len(rhythm_counts)

        # ② 多押分析 (10ms bin, 仅核心note)
        time_bins = {}
        for note in core_notes_v8:
            t_sec = time_to_seconds(note['time'], max(note.get('bpm', bpm), 1.0), bpm_timeline)
            t_bin = round(t_sec, 2)
            time_bins.setdefault(t_bin, []).append(note)
        chords = [len(g) for g in time_bins.values() if len(g) >= 2]
        features['avg_chord_size_poly'] = float(np.mean(chords)) if chords else 0.0
    else:
        features['fast_note_density_16th'] = 0.0
        features['rhythm_type_count'] = 0
        features['avg_chord_size_poly'] = 0.0

    return features


# ====== 底层工具函数 ======

def _compute_window_density(times, window_size):
    if times.size == 0:
        return np.array([0])
    max_t = float(times[-1])
    n_w = max(int(max_t / window_size) + 1, 1)
    bins = np.arange(0, (n_w + 1) * window_size, window_size)
    counts, _ = np.histogram(times, bins=bins)
    return counts


def _compute_high_duration(densities, threshold):
    if densities.size == 0 or threshold <= 0:
        return 0
    above = densities >= threshold
    if not np.any(above):
        return 0
    runs = np.diff(np.concatenate(([False], above, [False])).astype(int))
    run_starts = np.where(runs == 1)[0]
    run_ends = np.where(runs == -1)[0]
    lengths = run_ends - run_starts
    return int(np.max(lengths)) if len(lengths) > 0 else 0


def _compute_simultaneous_notes(notes):
    sim_th = 0.03125
    windows = defaultdict(list)
    for n in notes:
        tk = round(n['time'] / sim_th) * sim_th
        windows[tk].append(n)

    max_sim = 0
    total_sim = 0
    event_count = 0
    mf = {'count_3plus': 0, 'count_4plus': 0, 'max_simultaneous': 0}
    chord_sizes = {2: 0, 3: 0, 4: 0, 5: 0}
    pos_spreads = []
    weighted_mf_scores = []  # 加权多指协调分（区分流式/离散）
    discrete_mf_count = 0    # 离散型多指事件计数
    total_mf_events = 0      # 多指事件总数

    for tk, notes_in_window in windows.items():
        sz = len(notes_in_window)
        if sz > 1:
            max_sim = max(max_sim, sz)
            total_sim += sz
            event_count += 1
            if sz >= 3:
                mf['count_3plus'] += 1
                mf['max_simultaneous'] = max(mf['max_simultaneous'], sz)
            if sz >= 4:
                mf['count_4plus'] += 1
                mf['max_simultaneous'] = max(mf['max_simultaneous'], sz)
            key = min(sz, 5)
            chord_sizes[key] = chord_sizes.get(key, 0) + 1
            p = [n.get('positionX', 0) for n in notes_in_window]
            spread = max(p) - min(p)
            pos_spreads.append(spread)
            
            # 加权多指协调分：归一化跨度 = 总跨度 / (指头数-1)
            # 流式（1,2,3 → spread=2, sz=3 → norm=1.0）：简单
            # 离散（1,4,7 → spread=6, sz=3 → norm=3.0）：困难
            # 双指宽押（1,9 → spread=8, sz=2 → norm=8.0）：极难协调
            norm_spread = spread / max(sz - 1, 1)
            # 用sz加权：越大越多指越难，但是要用sqrt避免过度放大
            score = norm_spread * (sz ** 0.5)
            weighted_mf_scores.append(score)
            
            if sz >= 2:
                total_mf_events += 1
                if norm_spread > 1.5:  # 离散型：平均每个指头间距 > 1.5 轨道
                    discrete_mf_count += 1

    return {
        'max': max_sim, 'avg': total_sim / max(event_count, 1),
        'event_count': event_count, 'multi_finger_events': mf,
        'chord_sizes': chord_sizes, 'pos_spreads': pos_spreads,
        'weighted_mf_score_total': sum(weighted_mf_scores),
        'weighted_mf_score_mean': float(np.mean(weighted_mf_scores)) if weighted_mf_scores else 0,
        'discrete_mf_ratio': discrete_mf_count / max(total_mf_events, 1),
        'total_mf_events': total_mf_events,
    }


def _compute_concurrent_holds(all_notes):
    events = []
    for n in all_notes:
        if n['type'] == NOTE_HOLD:
            s = n['time']
            e = s + n.get('holdTime', 0)
            events.append((s, 1))
            events.append((e, -1))
    if not events:
        return {'max': 0, 'avg': 0, 'event_count': 0}

    events.sort(key=lambda x: (x[0], -x[1]))
    max_c = 0
    cur = 0
    total_c = 0
    cnt = 0
    for _, delta in events:
        cur += delta
        max_c = max(max_c, cur)
        total_c += cur
        cnt += 1
    return {'max': max_c, 'avg': total_c / max(cnt, 1), 'event_count': cnt}


def _compute_multifinger_bursts(notes):
    threshold = 0.03125
    windows = defaultdict(list)
    for n in notes:
        tk = round(n['time'] / threshold) * threshold
        windows[tk].append(n)

    sorted_t = sorted(windows.keys())
    segments = []
    cur_seg = []
    in_burst = False
    for t in sorted_t:
        if len(windows[t]) >= 3:
            if not in_burst:
                in_burst = True
                cur_seg = list(windows[t])
            else:
                cur_seg.extend(windows[t])
        else:
            if in_burst and cur_seg:
                segments.append(list(cur_seg))
                cur_seg = []
            in_burst = False
    if in_burst and cur_seg:
        segments.append(list(cur_seg))

    if segments:
        notes_per = [len(s) for s in segments]
        lengths = []
        for s in segments:
            if len(s) > 1:
                ts = [n['time'] for n in s]
                lengths.append(max(ts) - min(ts))
            else:
                lengths.append(0)
        return {
            'count': len(segments),
            'avg_notes': float(np.mean(notes_per)),
            'max_notes': max(notes_per),
            'avg_len': float(np.mean(lengths)),
            'max_len': max(lengths),
        }
    return {'count': 0, 'avg_notes': 0, 'max_notes': 0, 'avg_len': 0, 'max_len': 0}


def _compute_hold_interference_fast(all_notes, times, positions, hold_mask, n_hold, dt):
    if n_hold == 0 or len(times) < 2:
        return 0
    hold_start = times[hold_mask]
    hold_end = hold_start + np.array([n.get('holdTime', 0) for n in all_notes if n['type'] == NOTE_HOLD])
    hold_pos = positions[hold_mask]

    tap_flick = ~(np.array([n['type'] for n in all_notes]) == NOTE_DRAG)
    tap_flick = tap_flick & ~hold_mask
    tf_t = times[tap_flick]
    tf_p = positions[tap_flick]

    if len(tf_t) == 0:
        return 0

    total_interf = 0.0
    for hi in range(n_hold):
        left = np.searchsorted(tf_t, hold_start[hi], side='left')
        right = np.searchsorted(tf_t, hold_end[hi], side='right')
        if right > left:
            dists = np.abs(tf_p[left:right] - hold_pos[hi])
            total_interf += float(np.sum(dists))
    return total_interf / max(n_hold, 1)
