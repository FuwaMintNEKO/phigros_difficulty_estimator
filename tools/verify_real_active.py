# -*- coding: utf-8 -*-
"""验证 real_active 计算对稀疏EZ谱的异常行为 + Downloads自制谱格式分类"""
import os, sys, json, glob

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from feature_extractor import collect_all_notes, time_to_seconds, _parse_bpm_timeline

# ============ 1. Glaciaxion EZ real_active 根因 ============
print('='*70)
print('1. Glaciaxion EZ real_active 根因验证')
print('='*70)
path = os.path.join(_ROOT, 'data', 'chart', 'Glaciaxion.SunsetRay.0', 'EZ.json')
with open(path, 'r', encoding='utf-8') as f:
    cd = json.load(f)
all_notes, judge_lines, bpm_timeline = collect_all_notes(cd)
bpm = judge_lines[0].get('bpm', 120.0) if judge_lines else 120.0

n = len(all_notes)
times = sorted(n_['time'] for n_ in all_notes)
all_t_sec = sorted([time_to_seconds(t, max(n_['bpm'], 1.0), bpm_timeline) for t, n_ in zip(times, all_notes)])
gaps = [all_t_sec[i+1]-all_t_sec[i] for i in range(len(all_t_sec)-1)]
big_gaps = [g for g in gaps if g > 1.0]
print(f'notes={n}, 总跨度={all_t_sec[-1]-all_t_sec[0]:.2f}s')
print(f'gaps总数={len(gaps)}, >1s的大间隙={len(big_gaps)}个, 大间隙总和={sum(big_gaps):.2f}s')
print(f'real_active = 跨度 - 大间隙和 = {(all_t_sec[-1]-all_t_sec[0])-sum(big_gaps):.3f}s')
small_gaps = [g for g in gaps if g <= 1.0]
print(f'<=1s的小间隙总和 = {sum(small_gaps):.3f}s  (两者应相等)')
print(f'小间隙分布: count={len(small_gaps)}, 中位数={sorted(small_gaps)[len(small_gaps)//2]:.3f}s')
# 无bpm_timeline时的错误行为
no_tl = time_to_seconds(times[-1], bpm, None)
print(f'(对照: 无BPMList时 time_to_seconds 用恒定BPM={bpm}, 总跨度={no_tl:.2f}s)')

# ============ 2. Downloads 自制谱格式分类 ============
print()
print('='*70)
print('2. Downloads 自制谱格式分类')
print('='*70)
DL = r'C:\Users\NaNK\Downloads'
candidates = []
for ext in ('*.json', '*.mcz'):
    candidates += glob.glob(os.path.join(DL, ext))
# 过滤非谱面json（数字id开头的官谱、明显非谱面的）
def classify(p):
    fn = os.path.basename(p)
    if fn in ('0582581966828779.json','1145.json','1321664301929799.json','16032396.json',
              '1765944423.json','1770391855.json','202341.json','2155734445357448.json',
              '29834645.json','4641132726938698.json','51030697.json','61901444.json',
              '6923526264684294.json','7009367902368871.json','72443776.json','80116145.json',
              '81816997.json','82991127.json','93562988.json'):
        return '官方谱(数字ID)'
    try:
        with open(p, 'r', encoding='utf-8') as f:
            head = f.read(2000)
    except Exception as e:
        return f'读取失败: {e}'
    head_stripped = head.strip()
    if not head_stripped.startswith('{'):
        return 'PE文本'
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return f'JSON解析失败: {str(e)[:40]}'
    # RPE特征
    rpe = 'RPEVersion' in data.get('META', {})
    has_bpmlist = 'BPMList' in data
    jls = data.get('judgeLineList', [])
    # 判断notes字段方式
    note_storage = set()
    for jl in jls:
        if 'notes' in jl: note_storage.add('notes')
        if 'notesAbove' in jl or 'notesBelow' in jl: note_storage.add('above/below')
    ev_layers = sum(1 for jl in jls if 'eventLayers' in jl)
    fmt = 'RPE' if rpe else '官谱/标准'
    return (f'{fmt} | META={len(data.get("META",{}))} | BPMList={len(data.get("BPMList",[]))}'
            f' | 线={len(jls)} | 存储={sorted(note_storage)} | eventLayers线={ev_layers}')

print(f'共 {len(candidates)} 个候选文件')
for p in sorted(candidates):
    print(f'  {os.path.basename(p)[:60]:<62} -> {classify(p)}')

# ============ 3. PE 时间单位验证 ============
print()
print('='*70)
print('3. PE时间单位验证 (スタートリップ 12.2)')
print('='*70)
pe_path = os.path.join(DL, 'スタートリップ(12.2).json')
if os.path.exists(pe_path):
    with open(pe_path, 'r', encoding='utf-8') as f:
        text = f.read()
    max_n1 = 0.0; max_cp = 0.0; bpms = {}
    for raw in text.split('\n'):
        raw = raw.strip()
        if not raw or raw.startswith('#') or raw.startswith('&'):
            continue
        parts = raw.split()
        if not parts: continue
        cmd = parts[0]
        if cmd == 'n1' and len(parts) >= 3:
            max_n1 = max(max_n1, float(parts[2]))
        elif cmd == 'bp' and len(parts) >= 3:
            bpms[int(parts[1])] = float(parts[2])
        elif cmd == 'cp' and len(parts) >= 3:
            max_cp = max(max_cp, float(parts[2]))
    print(f'n1最大时间={max_n1:.1f}, cp最大时间={max_cp:.1f}, bp定义={bpms}')
    main_bpm = max(bpms.values()) if bpms else 0
    print(f'若单位是"拍": 总时长≈{max_n1/main_bpm*60:.1f}s ({max_n1}拍 @ {main_bpm}BPM)')
    print(f'若单位是"秒": 总时长≈{max_n1:.1f}s')
    # 歌词/song长度参照: 该曲目约3-4分钟
else:
    print('文件不存在')
