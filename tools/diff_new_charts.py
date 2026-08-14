# -*- coding: utf-8 -*-
"""对比新旧 difficulty.tsv 与 chart 目录, 找出:
  1. 新难度表里有但我们训练数据里没有的歌/谱面
  2. 定数有变动的歌
  3. 新增的高定数谱面
"""
import os, re

OLD_TSV = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator\data\info\difficulty.tsv'
NEW_TSV = r'D:\迅雷下载\Phigros_Resource-master\Phigros_Resource-master\info\difficulty.tsv'
OLD_CHART = r'd:\Trae项目\新建文件夹\phigros_difficulty_estimator\data\chart'
NEW_CHART = r'D:\迅雷下载\Phigros_Resource-master\Phigros_Resource-master\chart'

def load_tsv(path):
    """返回 {song_id: {level: 定数}}"""
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            sid = parts[0]
            vals = {}
            lv_names = ['EZ', 'HD', 'IN', 'AT']
            for i, v in enumerate(parts[1:]):
                if v.strip():
                    try:
                        vals[lv_names[i]] = float(v)
                    except ValueError:
                        pass
            out[sid] = vals
    return out

old_d = load_tsv(OLD_TSV)
new_d = load_tsv(NEW_TSV)

print(f'旧难度表歌曲数: {len(old_d)}, 新难度表歌曲数: {len(new_d)}')

# 新增歌曲
new_songs = sorted(set(new_d) - set(old_d))
print(f'\n=== 新增歌曲 ({len(new_songs)}) ===')
for s in new_songs:
    lv_str = ' | '.join(f'{lv}:{v}' for lv, v in new_d[s].items())
    print(f'  {s:<50} {lv_str}')

# 删除歌曲
removed = sorted(set(old_d) - set(new_d))
if removed:
    print(f'\n=== 新表没有的歌曲 ({len(removed)}) ===')
    for s in removed[:20]:
        print(f'  {s}')

# 定数变动
changed = []
for s in set(old_d) & set(new_d):
    if old_d[s] != new_d[s]:
        diff = [(lv, old_d[s].get(lv), new_d[s].get(lv)) for lv in ['EZ','HD','IN','AT'] if old_d[s].get(lv) != new_d[s].get(lv)]
        changed.append((s, diff))
print(f'\n=== 定数有变动的歌曲 ({len(changed)}) ===')
for s, diff in sorted(changed, key=lambda x: x[0]):
    dstr = ' | '.join(f'{lv}: {o}->{n}' for lv, o, n in diff)
    print(f'  {s:<50} {dstr}')

# chart 目录对比
old_dirs = set(os.listdir(OLD_CHART))
new_dirs = set(os.listdir(NEW_CHART))
missing_in_old = sorted(new_dirs - old_dirs)
extra_in_old = sorted(old_dirs - new_dirs)
print(f'\n=== 新chart目录有但旧没有的谱面目录 ({len(missing_in_old)}) ===')
for d in missing_in_old:
    files = os.listdir(os.path.join(NEW_CHART, d)) if os.path.isdir(os.path.join(NEW_CHART, d)) else []
    print(f'  {d}  ->  {files}')
print(f'\n=== 旧chart目录有但新没有的 ({len(extra_in_old)}) ===')
for d in extra_in_old[:30]:
    print(f'  {d}')
