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
        # 官谱: 顶层 speedEvents; RPE: eventLayers[*].speedEvents (顶层为空时用层内)
        top_events = line.get('speedEvents', [])
        layer_events = []
        for layer in line.get('eventLayers', []) or []:
            if layer is None:
                continue
            layer_events.extend(layer.get('speedEvents', []))
        events = top_events if top_events else layer_events
        for ev in events:
            if 'value' in ev:
                # 官谱格式: value = 倍率
                value = ev.get('value', 1.0)
                st = ev.get('startTime', 0)
                et = ev.get('endTime', 0)
            else:
                # RPE 格式: start/end = 每秒下降120px 的单位; 基准 5 = 官谱 1.0 倍率
                # (如 start=12 → 1440px/s → 2.4x), startTime/endTime 为 [m,b,d]
                value = ev.get('start', 5) / 5.0
                st = ev.get('startTime', [0, 0, 1])
                et = ev.get('endTime', [0, 0, 1])
            all_events.append({
                'line_idx': line_idx,
                'startTime': st,
                'endTime': et,
                'value': value,
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
        # 活跃时长 = 每个音符前后半阈值"气泡"的并集总长（扫描线法）
        # 稀疏谱中孤立音符各贡献1个阈值的活跃时间, 避免 real_active 塌缩导致密度特征爆炸
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

    # v11.8c: tap参与的击打密度hold也参与 (hold开始点=击打事件)
    t4 = _density_masked(tap_mask | hold_mask, 4)
    features['peak_tap_density_4beat'] = float(np.max(t4)) if t4.size > 0 else 0
    features['mean_tap_density_4beat'] = float(np.mean(t4)) if t4.size > 0 else 0

    # ====== 多押（仅 Tap + Hold；多押语义不含flick, 用户确认） ======
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
    # v11.6: 多押时间分布 (玩家视角: 多押密集段/间歇型 vs 全程型)
    _ev_times = simultaneous.get('event_times', [])
    _ev_bpms = simultaneous.get('event_bpms', [])
    if len(_ev_times) > 1:
        try:
            _ev_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_timeline) for t, b in zip(_ev_times, _ev_bpms)])
            _ds_eff = max(features['duration_sec'], 1.0)
            _nb1 = max(int(np.ceil(_ds_eff)), 1)
            _h1, _ = np.histogram(_ev_sec, bins=_nb1, range=(0, _ds_eff))
            features['chord_events_peak_1s'] = float(_h1.max()) if len(_h1) else 0.0
            _nb8 = max(int(np.ceil(_ds_eff / 8.0)), 1)
            _h8, _ = np.histogram(_ev_sec, bins=_nb8, range=(0, _ds_eff))
            features['chord_events_peak_8s'] = float(_h8.max()) if len(_h8) else 0.0
            features['chord_heavy_ratio_8s'] = float(np.mean(_h8 >= 20)) if len(_h8) else 0.0
        except Exception:
            features['chord_events_peak_1s'] = 0.0
            features['chord_events_peak_8s'] = 0.0
            features['chord_heavy_ratio_8s'] = 0.0
    else:
        features['chord_events_peak_1s'] = 0.0
        features['chord_events_peak_8s'] = 0.0
        features['chord_heavy_ratio_8s'] = 0.0

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

    # 和弦大小熵（和弦大小分布的复杂度）— v11修复: 标准香农熵 + 分布含单押(5类)
    single_ev = simultaneous.get('single_events', 0)
    counts = np.array([single_ev, cs.get(2, 0), cs.get(3, 0), cs.get(4, 0), cs.get(5, 0)], dtype=float)
    if counts.sum() > 0:
        cs_probs = counts / counts.sum()
        cs_probs = cs_probs[cs_probs > 0]
        # 标准香农熵: p>0已过滤, log2直接作用于p, 不加平滑项(避免 p->1 时负熵)
        ent = float(-np.sum(cs_probs * np.log2(cs_probs)))
        features['chord_size_entropy'] = ent
        # 归一化熵 [0,1] (5类分布最大熵 log2(5))
        features['chord_entropy_norm'] = float(ent / np.log2(5.0))
        # 多押复杂度: 熵 × 3+押事件占比 — 纯双押谱熵≈0且3+押少 → 接近0; 多押丰富且频繁 → 高
        mf_ratio_3p = (cs.get(3, 0) + cs.get(4, 0) + cs.get(5, 0)) / max(total_sim_ev, 1)
        features['chord_complexity'] = float(ent * mf_ratio_3p)
    else:
        features['chord_size_entropy'] = 0.0
        features['chord_entropy_norm'] = 0.0
        features['chord_complexity'] = 0.0

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
        # v11.10 锁手检测: hold长条按住期间的其他音符, 按难度加权 tap>flick>drag
        # 用户: tap(点击,锁手最难) > flick(划动,中等) > drag(零操作,几乎无难度)
        all_lock_tap = 0
        all_lock_flick = 0
        all_lock_drag = 0
        total_disp = 0.0
        max_disp = 0.0
        lock_t = times[tap_mask | flick_mask | drag_mask]
        lock_pos = positions[tap_mask | flick_mask | drag_mask]
        lock_types = types[tap_mask | flick_mask | drag_mask]
        lock_sorted_idx = np.argsort(lock_t)
        lock_t_sorted = lock_t[lock_sorted_idx]
        lock_pos_sorted = lock_pos[lock_sorted_idx]
        lock_types_sorted = lock_types[lock_sorted_idx]

        for hi in range(n_hold):
            left = np.searchsorted(lock_t_sorted, hold_t[hi], side='left')
            right = np.searchsorted(lock_t_sorted, hold_end[hi], side='right')
            if right > left:
                for j in range(left, right):
                    t_ = lock_types_sorted[j]
                    if t_ == NOTE_TAP: all_lock_tap += 1
                    elif t_ == NOTE_FLICK: all_lock_flick += 1
                    elif t_ == NOTE_DRAG: all_lock_drag += 1
                disps = np.abs(lock_pos_sorted[left:right] - hold_pos[hi])
                total_disp += float(np.sum(disps))
                max_disp = max(max_disp, float(np.max(disps)))

        features['hold_lock_tap_events'] = all_lock_tap
        features['hold_lock_tap_events_per_hold'] = all_lock_tap / max(n_hold, 1)
        # v11.10: 加权锁手分 (tap×1.0 + flick×0.4 + drag×0.1) — 反映锁手实际难度贡献
        lock_weighted = all_lock_tap * 1.0 + all_lock_flick * 0.4 + all_lock_drag * 0.1
        features['hold_lock_weighted'] = lock_weighted
        features['hold_lock_weighted_per_hold'] = lock_weighted / max(n_hold, 1)
        features['hold_lock_weighted_per_sec'] = lock_weighted / max(ds, 0.01)
        features['hold_lock_flick_events'] = all_lock_flick
        features['hold_lock_drag_events'] = all_lock_drag
        features['hold_lock_avg_displacement'] = total_disp / max(all_lock_tap + all_lock_flick + all_lock_drag, 1)
        features['hold_lock_max_displacement'] = max_disp
        features['hold_lock_displacement_per_sec'] = total_disp / ds
    else:
        features.update({'hold_lock_tap_events': 0, 'hold_lock_tap_events_per_hold': 0,
                         'hold_lock_weighted': 0, 'hold_lock_weighted_per_hold': 0, 'hold_lock_weighted_per_sec': 0,
                         'hold_lock_flick_events': 0, 'hold_lock_drag_events': 0,
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
        d_tap = _density_masked(tap_mask | hold_mask, mw)   # v11.8c: 含hold
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
    tap_d1 = _density_masked(tap_mask | hold_mask, 1)   # v11.8c: 含hold
    if tap_d1.size > 2:
        td_mean = float(np.mean(tap_d1))
        td_max = float(np.max(tap_d1))
        td_sorted = np.sort(tap_d1)[::-1]
        top5_n = max(5, len(tap_d1) // 20)
        features['tap_burst_peak_to_mean'] = td_max / max(td_mean, 0.01)
        features['tap_burst_top5'] = float(np.mean(td_sorted[:top5_n]))
        p95 = float(np.percentile(tap_d1, 95))
        features['extreme_tap_window_ratio'] = float(np.sum(tap_d1 >= p95) / max(len(tap_d1), 1))

    tap_d05 = _density_masked(tap_mask | hold_mask, 0.5)   # v11.8c: 含hold
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
    # v11.7: 长黄键(drag带holdTime, RPE特有) — Feeling Blue类全drag长条谱曾完全失明
    drag_ht = hold_times[drag_mask] if n_notes > 0 else np.array([])
    features['drag_hold_count'] = int(np.sum(drag_ht > 0))
    features['drag_hold_time_total_beats'] = float(np.sum(drag_ht))
    features['drag_hold_time_total_sec'] = time_to_seconds(float(np.sum(drag_ht)), bpm)
    features['drag_hold_avg_beats'] = float(np.mean(drag_ht[drag_ht > 0])) if np.any(drag_ht > 0) else 0.0
    features['drag_hold_ratio'] = float(np.sum(drag_ht > 0)) / max(n_drag, 1)
    # v11.7b: 纯drag滑动谱 — drag击打密度 (Feeling Blue 820 drag/117s = 7/s)
    features['drag_per_sec'] = n_drag / max(features['duration_sec'], 0.01)

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
    # FIX(2026-08-13): 原 gaps<=4(4ticks≈0.125拍) 阈值过严, 238BPM交互的相邻音符间隔63ms被全过滤
    # → Verrückt 大位移交互的 movement_per_second 几乎为0。改为秒级阈值(0.5s), 用相邻音符平均BPM换算。
    # FIX: cross_hand 原用 ticks 当秒(<0.25永远不成立) → 用真实秒间隔。
    # v11.8c: 位移含hold长条 (hold开始点参与位移/跨线/换手; 用户实测Feeling Blue全hold位移1375被完全忽略)
    active_mask = tap_mask | flick_mask | hold_mask
    active_idx = np.where(active_mask)[0]
    active_t = times[active_mask]
    active_pos = positions[active_mask]
    if len(active_t) > 1:
        bpm_pair = (note_bpms[active_idx[:-1]] + note_bpms[active_idx[1:]]) / 2.0
        gaps = np.abs(np.diff(active_t))
        # v11.15修复: 间隔秒 = tick/32*60/bpm (局部bpm), 不用积分
        gaps_sec = gaps / 32.0 * 60.0 / np.maximum(bpm_pair, 1.0)
        pos_diffs = np.abs(np.diff(active_pos))
        valid = gaps_sec <= 0.5
        distances = pos_diffs[valid]
        features['avg_movement'] = float(np.mean(distances)) if len(distances) > 0 else 0
        features['total_movement'] = float(np.sum(distances)) if len(distances) > 0 else 0
        features['max_movement'] = float(np.max(distances)) if len(distances) > 0 else 0
        features['movement_per_second'] = features['total_movement'] / ds
        # cross_hand: 相邻tap间隔<0.25s 且 X位移>3格
        cross_hand_count = 0
        for i in range(1, len(active_t)):
            if gaps_sec[i-1] < 0.25 and pos_diffs[i-1] > 3.0:
                cross_hand_count += 1
        features['cross_hand_density'] = cross_hand_count / max(ds, 0.01)
        # v11.5: 跨线切换 (lane switch): 相邻击打位置跨lane(>1.5格), 含连续跨线run检测
        lane_switch = pos_diffs > 1.5
        features['lane_switch_count'] = int(np.sum(lane_switch))
        features['lane_switch_ratio'] = float(np.mean(lane_switch)) if len(lane_switch) > 0 else 0.0
        features['lane_switch_density'] = features['lane_switch_count'] / max(ds, 0.01)
        # 连续跨线run (>=2连续跨线 = 跨线穿梭段)
        if len(lane_switch) > 0:
            ls_int = lane_switch.astype(int)
            runs = np.diff(np.concatenate(([0], ls_int, [0])))
            starts = np.where(runs == 1)[0]
            ends = np.where(runs == -1)[0]
            run_lens = ends - starts
            features['crossline_chain_max'] = int(np.max(run_lens)) if len(run_lens) > 0 else 0
            features['crossline_chain_ratio'] = float(np.sum(run_lens >= 3)) / max(len(lane_switch), 1)
        else:
            features['crossline_chain_max'] = 0
            features['crossline_chain_ratio'] = 0.0
    else:
        features.update({'avg_movement': 0, 'total_movement': 0, 'max_movement': 0, 'movement_per_second': 0,
                         'cross_hand_density': 0, 'lane_switch_count': 0, 'lane_switch_ratio': 0.0,
                         'lane_switch_density': 0.0, 'crossline_chain_max': 0, 'crossline_chain_ratio': 0.0})
    # 位移×密度复合: 大位移交互(238bpm长段位移) 的位移强度随密度放大 (Verrückt 底力门槛)
    features['movement_density_index'] = features.get('movement_per_second', 0) * features.get('real_core_notes_per_second', 0)

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
    # speed_volatility: 流速忽慢忽快程度（读谱干扰）
    features['speed_volatility'] = features.get('speed_std', 0) * features['speed_event_density']
    # v11.5: log压缩 (树模型对OOD极端值平台化, log可外推)
    features['speed_event_log_density'] = float(np.log1p(features['speed_event_density']))
    features['speed_volatility_log'] = float(np.log1p(features['speed_volatility']))

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
    core_idx_1s = np.where(core_mask_1s)[0]  # 全局音符索引 (用于取正确BPM)
    core_notes_times = times[core_mask_1s]
    if len(core_notes_times) > 5:
        core_t_sec_1s = np.array([time_to_seconds(t, max(all_notes[idx].get('bpm', bpm), 1.0), bpm_timeline)
                                   for idx, t in zip(core_idx_1s, core_notes_times)])
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

    # ====== 有效单指密度 (同押去冗余: 1秒窗口内"独立击打次数", 多指全押只算1次) ======
    # 背景: core_peak_density_1sec 不区分"4押全押"(多指顺手, 如volcanic中间段 4指打8分全押)
    #   vs "单指连打"(键盘底力, 如D321/梦降日)。两者 core_peak_1sec 几乎相同但难度天差地别。
    # 解法: 窗口内音符按 tick 聚类(同押组 tick 差 < 1), 组数 = 有效独立击打次数。
    #   volcanic 4押海 1秒28音符 → 有效≈7; 键盘连打 1秒27单点 → 有效=27。
    if len(core_notes_times) > 5:
        core_t_sec_1s = np.array([time_to_seconds(t, max(all_notes[idx].get('bpm', bpm), 1.0), bpm_timeline)
                                   for idx, t in zip(core_idx_1s, core_notes_times)])
        order = np.argsort(core_t_sec_1s)
        cts_sorted = core_t_sec_1s[order]
        ctk_sorted = core_notes_times[order]
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
        features['eff_peak_tps_1s'] = int(max_eff)
        features['eff_avg_tps_1s'] = float(np.mean(eff_vals))
    else:
        features['eff_peak_tps_1s'] = 0
        features['eff_avg_tps_1s'] = 0.0

    # ====== 密度维度：√(真实TPS × 高潮段平均TPS) ======
    # v11.2: 高潮段TPS改用"有效击打数"(同押去冗余, 4k全押1窗口计1次), 修复多押撑密度虚高
    # 依据: t2研究 — 官谱corr保持(P 0.904/ S 0.948), 上架谱偏差 +0.040→-0.006, 高难段降8-18%
    rcnps = features.get('real_core_notes_per_second', 0)
    above_avg_mean = rcnps  # fallback
    above_avg_ratio = 0.0
    above_avg_dur = 0
    if len(core_times) > 5:
        core_idx_all = np.where(core_mask)[0]  # 全局音符索引 (用于取正确BPM)
        t_arr = np.sort(np.array([time_to_seconds(t, max(all_notes[idx].get('bpm', bpm), 1.0), bpm_timeline)
                                   for idx, t in zip(core_idx_all, core_times)]))
        if len(core_notes_times) > 5:
            # 复用 eff 块的排序 (cts_sorted=秒, ctk_sorted=tick)
            eff_ref = float(np.mean(eff_vals)) if eff_vals else rcnps  # eff版阈值
            left = 0; above_windows = []; above_windows_eff = []
            for right in range(len(cts_sorted)):
                while cts_sorted[right] - cts_sorted[left] > 1.0:
                    left += 1
                window_tps = right - left + 1
                seg_tick = ctk_sorted[left:right + 1]
                if len(seg_tick) >= 2:
                    eff_count = 1 + int(np.sum(np.diff(seg_tick) >= 1))
                else:
                    eff_count = int(len(seg_tick))
                if window_tps >= rcnps:
                    above_windows.append(window_tps)
                if eff_count >= eff_ref:
                    above_windows_eff.append(eff_count)
            above_avg_mean = float(np.mean(above_windows)) if above_windows else rcnps
            above_avg_mean_eff = float(np.mean(above_windows_eff)) if above_windows_eff else above_avg_mean
            above_avg_ratio = len(above_windows) / max(len(t_arr), 1)
            above_avg_dur = len(above_windows)
        else:
            left = 0; above_windows = []
            for right in range(len(t_arr)):
                while t_arr[right] - t_arr[left] > 1.0:
                    left += 1
                window_tps = right - left + 1
                if window_tps >= rcnps:
                    above_windows.append(window_tps)
            above_avg_mean = float(np.mean(above_windows)) if above_windows else rcnps
            above_avg_mean_eff = above_avg_mean
            above_avg_ratio = len(above_windows) / max(len(t_arr), 1)
            above_avg_dur = len(above_windows)
    else:
        above_avg_mean_eff = rcnps
    features['above_avg_density_mean'] = above_avg_mean_eff  # v11.2: 有效击打版
    features['above_avg_density_ratio'] = above_avg_ratio
    features['above_avg_duration_sec'] = above_avg_dur  # 高潮段总持续秒数
    features['density_dimension'] = float(np.sqrt(max(rcnps, 0.01) * max(above_avg_mean_eff, 0.01)))

    # ====== 耐力指标(保留供GB模型219特征使用，不在FLAT_FEATURES中) ======
    stamina_mask = tap_mask | hold_mask
    core_t = times[stamina_mask]
    if len(core_t) > 10:
        core_bpm_arr = np.array([n.get('bpm', bpm) for n in all_notes])[stamina_mask]
        core_t_sec = np.array([time_to_seconds(t, max(b, 1.0), bpm_timeline) for t, b in zip(core_t, core_bpm_arr)])
        core_t_sec.sort()
        avg_core_tps = len(core_t_sec) / max(ds, 0.01)
        threshold = avg_core_tps * 0.9
        left = 0; high_sec = 0; total_windows = 0
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
        # v11.15修复: 间隔秒数 = tick/32 * 60/bpm(局部bpm), 不能用积分(积分返回累计绝对时间)
        intervals_sec = intervals / 32.0 * 60.0 / np.maximum(bpm_arr[1:], 1.0)

        # global: 极短间隔密度指标 — v11.14: 只算tap+hold的相邻间隔, 排除drag/flick; 排除多押(0ms)
        # 用户底线: 分音(16分/24分)只由tap+hold构成; drag(零操作)/flick不算; 0ms=多押非连续音符
        core_adj = (core_mask[1:] & core_mask[:-1]) & (intervals_sec > 1e-6)
        features['global_jack_count'] = int(np.sum(core_adj & (intervals_sec < 0.125)))
        features['miniburst_count'] = int(np.sum(core_adj & (intervals_sec < 0.0625)))
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

        # v11.10: 速度归一化分档 (用户: 100bpm的32分=200bpm的16分, 按真实毫秒间隔而非拍)
        # 拍域32分检测已删除 (thirtysecond_run_*: 用户确认没必要, BPM归一化即可)
        pos_diffs_arr = np.abs(np.diff(positions))
        same_line_mask = jl_idx[1:] == jl_idx[:-1]
        its_full = intervals_sec
        # v11.14: 只算tap+hold且排除多押(0ms); 同线限定保留 (跨线交错非手指速度)
        core_adj_full = (core_mask[1:] & core_mask[:-1]) & (intervals_sec > 1e-6)
        its = intervals_sec[same_line_mask & core_adj_full]
        features['fast_ms_050_ratio'] = float(np.sum(its < 0.05)) / max(len(its), 1)
        features['fast_ms_100_ratio'] = float(np.sum((its >= 0.05) & (its < 0.10))) / max(len(its), 1)
        features['fast_ms_150_ratio'] = float(np.sum((its >= 0.10) & (its < 0.15))) / max(len(its), 1)
        # 速度归一化交互段: 同线间隔<100ms且位置交替的连续run
        itr = np.zeros(n_notes, dtype=bool)
        for i in range(1, n_notes):
            if its_full[i-1] < 0.10 and same_line_mask[i-1] and pos_diffs_arr[i-1] > 1.5:
                itr[i] = True
                itr[i-1] = True
        mrun = np.diff(np.concatenate(([0], itr.astype(int), [0])))
        mst = np.where(mrun == 1)[0]; mend = np.where(mrun == -1)[0]
        mlens = mend - mst
        features['interaction_ms_run_max'] = int(np.max(mlens)) if len(mlens) > 0 else 0
        features['interaction_ms_run_ratio'] = float(np.sum(itr)) / max(n_notes, 1)

        tempo_change = 0
        for i in range(1, len(intervals)):
            if intervals[i] > intervals[i-1] * 1.5 or intervals[i] < intervals[i-1] * 0.67:
                tempo_change += 1
        features['tempo_change_count'] = tempo_change
        features['tempo_change_ratio'] = tempo_change / max(len(intervals), 1)
        # v11.5: 节奏突变密度 log压缩 (变速欺诈谱: 密度高但变速频繁)
        features['tempo_change_log_density'] = float(np.log1p(tempo_change / max(ds, 0.01)))

    # ====== 音符级差速 (note speed 字段, 默认1.0; 官谱/RPE音符均可能携带) ======
    # 差速 = 按键流速变化 (Retribution 全程差速, 含 speed=500/2000 极端值)
    note_speeds = np.array([float(n.get('speed', 1.0) or 1.0) for n in all_notes])
    log_speeds = np.log2(np.maximum(note_speeds, 1.0))
    speed_non1 = note_speeds != 1.0
    features['note_speed_non1_count'] = int(np.sum(speed_non1))
    features['note_speed_non1_ratio'] = float(np.mean(speed_non1)) if n_notes else 0.0
    features['note_speed_std'] = float(np.std(log_speeds)) if n_notes else 0.0
    features['note_speed_max'] = float(np.max(log_speeds)) if n_notes else 0.0
    features['note_speed_density'] = float(np.sum(speed_non1)) / max(features['duration_sec'], 0.01)
    # 高流速长条 (speed>=2 的 hold) = 闪现长条近似: 长条瞬间扫过屏幕
    fast_hold = (types == 3) & (note_speeds >= 2.0)
    features['fast_hold_count'] = int(np.sum(fast_hold))
    features['fast_hold_ratio'] = float(np.sum(fast_hold)) / max(np.sum(types == 3), 1)

    # ====== visibleTime 闪现 (官谱 hold 音符的提前显示时间, 默认999999) ======
    # 闪现 = 显示时间极短, 读谱压力大 (闪条/闪现长条)
    vts = np.array([float(n.get('visibleTime', 999999.0) or 999999.0) for n in all_notes])
    flash = vts < 999999.0
    features['flash_note_count'] = int(np.sum(flash))
    features['flash_note_ratio'] = float(np.mean(flash)) if n_notes else 0.0
    flash_hold = (types == 3) & flash
    features['flash_hold_count'] = int(np.sum(flash_hold))
    features['flash_hold_ratio'] = float(np.sum(flash_hold)) / max(np.sum(types == 3), 1)
    features['visible_time_min'] = float(np.min(vts)) if n_notes else 999999.0

    # ====== 和弦重键 (chord jack: 同线连续和弦快速重复, 重键4k/尾杀密集2k) ======
    # 分组: 同一判定线, 时间间隔 < 4 ticks(约50ms@120BPM) 的音符合并为同一和弦事件
    chord_jack_steps = 0
    chord_jack_3plus_pairs = 0
    if n_notes > 3:
        jl_idx_arr = np.array([n.get('judge_line_idx', 0) for n in all_notes])
        cur_group = -1
        prev_t = None
        g_sizes, g_times, g_lines, g_bpms = [], [], [], []
        for i in range(n_notes):
            if prev_t is None or (times[i] - prev_t) >= 4:
                cur_group += 1
                g_times.append(float(times[i]))
                g_lines.append(int(jl_idx_arr[i]))
                g_sizes.append(1)
                g_bpms.append(float(note_bpms[i]))
            else:
                g_sizes[cur_group] += 1
            prev_t = times[i]
        for g in range(1, len(g_sizes)):
            if (g_lines[g] == g_lines[g-1] and g_sizes[g] >= 2 and g_sizes[g-1] >= 2):
                gap_sec = (g_times[g] - g_times[g-1]) / 32.0 * 60.0 / max(g_bpms[g-1], 1.0)  # v11.15: 局部bpm直算
                if gap_sec < 0.12:
                    chord_jack_steps += 1
                    if g_sizes[g] >= 3 and g_sizes[g-1] >= 3:
                        chord_jack_3plus_pairs += 1
    features['chord_jack_steps'] = chord_jack_steps
    features['chord_jack_density'] = chord_jack_steps / max(features['duration_sec'], 0.01)
    features['chord_jack_3plus_pairs'] = chord_jack_3plus_pairs

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
    act_mask = tap_mask | flick_mask | hold_mask   # v11.8c: 含hold
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
        features['burst_movement_variance'] = float(np.var(burst_moves)) if burst_moves else 0
    else:
        features.update({'left_ratio': 0, 'right_ratio': 0, 'center_ratio': 0, 'spread_balance': 0,
                         'burst_avg_movement': 0, 'burst_max_movement': 0, 'burst_movement_ratio': 0,
                         'burst_movement_variance': 0})

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

    # ====== 读谱: Phigros判定线视觉干扰 ======
    # 兼容标准格式(judgeLineMoveEvents)和RPE格式(eventLayers)
    jline_move_total = 0; jline_rotate_total = 0; jline_disappear_total = 0
    has_above_notes = False; has_below_notes = False
    for line in judge_lines:
        # 标准格式: 顶层事件
        jline_move_total += len(line.get('judgeLineMoveEvents', []))
        jline_rotate_total += len(line.get('judgeLineRotateEvents', []))
        jline_disappear_total += len(line.get('judgeLineDisappearEvents', []))
        # RPE格式: eventLayers 嵌套事件
        for layer in line.get('eventLayers', []):
            if layer is None: continue
            jline_move_total += len(layer.get('moveXEvents', [])) + len(layer.get('moveYEvents', []))
            jline_rotate_total += len(layer.get('rotateEvents', []))
        # RPE格式: extended 附加事件
        ext = line.get('extended', {})
        jline_move_total += len(ext.get('inclineEvents', []))  # 倾角变化≈移动
        # above/below notes (RPE v3直接用notes, 标准用notesAbove/notesBelow)
        if line.get('notesAbove'): has_above_notes = True
        if line.get('notesBelow'): has_below_notes = True
        # RPE v3: notes数组本身就有above/below混合
        na_raw = line.get('notesAbove', None)
        nb_raw = line.get('notesBelow', None)
        has_above_notes = has_above_notes or (na_raw is not None)
        has_below_notes = has_below_notes or (nb_raw is not None)
    features['jline_movement_density'] = jline_move_total / max(ds, 0.01)
    features['jline_rotate_density'] = jline_rotate_total / max(ds, 0.01)
    features['jline_disappear_density'] = jline_disappear_total / max(ds, 0.01)
    # v11.5: 欺诈跨线比 — 判定线几乎不动但音符跨线密集 (3rd Avenue类: 视觉无引导手指要跨)
    features['jline_relative_cross'] = features.get('cross_hand_density', 0) / max(features['jline_movement_density'], 1.0)
    features['above_below_cross'] = 1.0 if has_above_notes and has_below_notes else 0.0

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
        wide = int(np.sum((time_gaps < 0.25) & (pos_gaps > 2.5)))
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
        tap_only = times[tap_mask | hold_mask]   # v11.8c: 含hold
        tap_pos = positions[tap_mask | hold_mask]
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

    # ====== 对拍/对切（连续多押事件快速交替 = 双手轮指打多押, kyou"多指-对拍/对切"） ======
    # 对拍: 双手交替快速击打多押; 对切: 双手同步/镜像连续轮指。核心是"多押→多押"间隔短。
    # 注意: core_windows 的 key 是 ticks(1/32拍单位), 0.5拍 = 16 ticks。
    if len(core_windows) > 2:
        chord_ev_times = sorted(tk for tk, notes in core_windows.items() if len(notes) >= 2)
        cc_alt = 0
        for i in range(1, len(chord_ev_times)):
            if chord_ev_times[i] - chord_ev_times[i-1] <= 16:  # 间隔 <= 0.5拍 (8分对拍)
                cc_alt += 1
        features['chord_chord_alt_count'] = cc_alt
        features['chord_chord_alt_rate'] = cc_alt / max(ds, 0.01)
    else:
        features['chord_chord_alt_count'] = 0
        features['chord_chord_alt_rate'] = 0.0

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
        fast_16th = 0; fast_32nd = 0; fast_24th = 0; fast_48th = 0; fast_64th = 0; rhythm_counts = {}
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
                                      (24,1/24),(28,1/28),(32,0.03125),
                                      (48,1/48),(64,1/64)]:
                    if abs(beats_val - target) / max(target, 0.001) < 0.12:
                        matched = frac; break
                if matched:
                    if matched >= 4: fast_16th += 1
                    if matched >= 8: fast_32nd += 1
                    if matched >= 12: fast_24th += 1
                    if matched >= 16: fast_48th += 1
                    if matched >= 32: fast_64th += 1
                    if matched >= 2: rhythm_counts[matched] = rhythm_counts.get(matched, 0) + 1
                    break
                elif beats_val > 0.02:
                    rhythm_counts[0] = rhythm_counts.get(0, 0) + 1; break
        features['fast_note_density_16th'] = fast_16th / max(dur, 0.01)
        features['fast_note_density_32nd'] = fast_32nd / max(dur, 0.01)
        features['fast_note_density_24th'] = fast_24th / max(dur, 0.01)
        features['fast_note_density_48th'] = fast_48th / max(dur, 0.01)
        features['fast_note_density_64th'] = fast_64th / max(dur, 0.01)
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
        features['fast_note_density_32nd'] = 0.0
        features['fast_note_density_24th'] = 0.0
        features['fast_note_density_48th'] = 0.0
        features['fast_note_density_64th'] = 0.0
        features['rhythm_type_count'] = 0
        features['avg_chord_size_poly'] = 0.0

    # 尾杀特征 (末段集中度, 社区"尾杀拉高定数"共识)
    features.update(compute_tail_features(all_notes, judge_lines, bpm_timeline, fallback_bpm))
    # v11.1: 定轨键盘段特征 (4k/5k/6k: 固定槽位密集击打, 多指分工, 双指无解)
    features.update(compute_track_segments_features(times, positions, bpm))

    return features


def compute_track_segments_features(times, positions, fallback_bpm):
    """定轨键盘段检测 (v11.1): 滑动窗口内 positionX 聚成的主槽位数 k (4k/5k/6k)
    - 窗口: 2.5秒 (密集段才统计: 窗口音符>=6)
    - 聚类: positionX 排序后间距>=1.5 分隔为不同槽位
    - 主导槽: 槽内音符>=4 (排除边缘噪声)
    - 输出: 4+/5+/6+槽位段的时长、最大槽位数、活跃段平均槽位数
    """
    out = {'tracks_4plus_sec': 0.0, 'tracks_5plus_sec': 0.0, 'tracks_6plus_sec': 0.0,
           'tracks_max_k': 0.0, 'tracks_avg_k': 0.0, 'tracks_active_sec': 0.0}
    n = len(times)
    if n < 6:
        return out
    bpm0 = fallback_bpm if fallback_bpm else 120.0
    secs = np.array(times, dtype=float) * 60.0 / (32.0 * bpm0)  # tick -> 秒
    WIN = 2.5  # 秒
    t0, t1 = float(secs[0]), float(secs[-1])
    if t1 - t0 < WIN:
        return out
    ks = []
    active_sec = 0.0
    cur = t0
    while cur < t1 - 1e-9:
        m = (secs >= cur) & (secs < cur + WIN)
        cnt = int(m.sum())
        if cnt >= 6:
            ps = sorted(set(np.round(positions[m] * 2) / 2))
            # 间距>=1.5 聚类
            clusters = []
            c = [ps[0]]
            for p in ps[1:]:
                if p - c[-1] < 1.5:
                    c.append(p)
                else:
                    clusters.append(c); c = [p]
            clusters.append(c)
            main_k = 0
            for cl in clusters:
                # 槽内音符数
                cl_cnt = sum(1 for p in positions[m] if any(abs(p - x) < 0.75 for x in cl))
                if cl_cnt >= 4:
                    main_k += 1
            if main_k >= 1:
                ks.append(main_k)
                active_sec += WIN
        cur += WIN
    if not ks:
        return out
    ks_arr = np.array(ks)
    out['tracks_4plus_sec'] = float((ks_arr >= 4).sum() * WIN)
    out['tracks_5plus_sec'] = float((ks_arr >= 5).sum() * WIN)
    out['tracks_6plus_sec'] = float((ks_arr >= 6).sum() * WIN)
    out['tracks_max_k'] = float(ks_arr.max())
    out['tracks_avg_k'] = float(ks_arr.mean())
    out['tracks_active_sec'] = float(active_sec)
    return out


def compute_tail_features(all_notes, judge_lines, bpm_timeline, fallback_bpm):
    """尾杀特征: 末段(最后15%时长)的密度集中度与1秒峰值
    社区证据: DF AT 最后5秒"全游最难尾杀"; QZKago 尾杀20秒"拉高一大档"。
    """
    if not all_notes:
        return {}
    times = np.array([n['time'] for n in all_notes])
    types = np.array([n['type'] for n in all_notes])
    tsec = np.array([time_to_seconds(t, max(n.get('bpm', fallback_bpm), 1.0), bpm_timeline)
                     for t, n in zip(times, all_notes)])
    total_sec = _compute_duration_sec(bpm_timeline, times[-1] / 32.0)
    if total_sec <= 0:
        return {}
    core = (types == NOTE_TAP) | (types == NOTE_HOLD)
    cut = total_sec * 0.85
    tail_mask = tsec >= cut
    tail_core = tsec[tail_mask & core]
    out = {'tail_note_count': float(tail_mask.sum()),
           'tail_ratio': float(tail_mask.sum() / len(tsec))}
    if tail_core.size > 3 and (total_sec - cut) > 0.5:
        # 末段1秒滑动窗口峰值 (简化为窗口直方图)
        win = 1.0
        nb = max(int(np.ceil((total_sec - cut) / win)), 1)
        counts = np.zeros(nb)
        for t in tail_core:
            idx = int((t - cut) / win)
            if 0 <= idx < nb:
                counts[idx] += 1
        tail_peak_1s = float(counts.max())
        # 全局1秒峰值
        nb_all = max(int(np.ceil(total_sec / win)), 1)
        all_counts = np.zeros(nb_all)
        for t in tsec[core]:
            idx = int(t / win)
            if 0 <= idx < nb_all:
                all_counts[idx] += 1
        global_peak_1s = float(all_counts.max()) if all_counts.max() > 0 else 1.0
        global_mean_1s = float(np.mean(all_counts)) if all_counts.size else 0.0
        out['tail_peak_1s_ratio'] = tail_peak_1s / max(global_peak_1s, 1.0)  # 末段峰值/全局峰值
        out['tail_peak_vs_mean'] = tail_peak_1s / max(global_mean_1s, 0.01)  # 末段峰值/全局均值
        out['tail_density'] = float(tail_core.size / max(total_sec - cut, 0.01))  # 末段核心密度
        out['tail_core_share'] = float(tail_core.size / max(core.sum(), 1))  # 末段核心音符占比
    return out


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

    event_times = []   # v11.6: 和弦事件时间序列 (多押分布特征)
    event_bpms = []
    for tk, notes_in_window in windows.items():
        sz = len(notes_in_window)
        if sz > 1:
            event_times.append(tk)
            event_bpms.append(notes_in_window[0].get('bpm', 120.0))
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
        'single_events': len(windows) - event_count,   # 单押窗口数 (v11修复: 熵分布需含单押)
        'event_times': event_times,   # v11.6
        'event_bpms': event_bpms,     # v11.6
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
