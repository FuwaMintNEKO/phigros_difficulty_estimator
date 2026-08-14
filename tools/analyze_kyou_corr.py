# -*- coding: utf-8 -*-
"""kyou 特征投票 × 模型 OOF 残差分析

问题1: 投票特征是否被模型吃透?
  - 对每个标签, 比较"被投该标签" vs "未投该标签" 的 OOF 残差 (err = oof - true)
  - 若某标签组残差系统性为负(低估), 说明模型没捕捉该特征 → 找新特征方向
问题2: 我们的特征与投票是否相关? (验证特征有效性)
  - 每个标签 vs 候选特征 的 Spearman 相关

依赖: tools/_oof_rows.pkl (由 analyze_oof_bias.py 生成)
"""
import os, sys, json, re, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
TOOLS = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from scipy.stats import spearmanr


def norm(s):
    return re.sub(r'[^a-z0-9\u4e00-\u9fff]+', '', str(s).lower())


def load_votes():
    votes = json.load(open(os.path.join(_ROOT, 'data', 'kyou', 'votes.json'), encoding='utf-8'))
    songlist = json.load(open(os.path.join(_ROOT, 'data', 'kyou', 'songlist.json'), encoding='utf-8'))['data']
    title2id = {}
    for s in songlist:
        title2id.setdefault(norm(s['标题']), s['id'])
    return votes, title2id


def main():
    rows = pickle.load(open(os.path.join(TOOLS, '_oof_rows.pkl'), 'rb'))
    names, levels, y, oof, errs, feats = rows['names'], rows['levels'], rows['y'], rows['oof'], rows['errs'], rows['feats']
    votes, title2id = load_votes()

    # 标签树
    tree = json.load(open(os.path.join(_ROOT, 'data', 'kyou', 'tags_tree.json'), encoding='utf-8'))
    id2name = {}
    for t in tree:
        id2name[t['id']] = t['name']
        for c in t.get('children', []):
            id2name[c['id']] = c['name']

    # 匹配: folder -> 标题 -> kyou id
    n = len(names)
    kyou_id = [None] * n
    for i, fn in enumerate(names):
        base = fn[:-2] if fn.endswith('.0') else fn
        t = norm(base.rsplit('.', 1)[0])
        kyou_id[i] = title2id.get(t)

    # 每行: (idx, level) -> tag votes dict
    lv2key = {'EZ': 'ez', 'HD': 'hd', 'IN': 'in', 'AT': 'at'}
    row_votes = [None] * n
    for i in range(n):
        kid, lv = kyou_id[i], lv2key.get(levels[i])
        if not kid or not lv:
            continue
        entry = votes.get(f'{kid}::{lv}')
        if entry and entry.get('topTags'):
            row_votes[i] = {t['tagName']: t['voteCount'] for t in entry['topTags']}

    matched = sum(1 for v in row_votes if v)
    print(f'OOF样本 {n}, 匹配到投票 {matched}')

    # ===== 问题1: 每个标签的残差对比 =====
    print('\n===== 投票标签 × OOF残差 (err = OOF预测 - 真实) =====')
    print(f'{"标签":<10} {"被投n":>5} {"未投n":>5} {"被投均残差":>10} {"未投均残差":>10} {"差值":>8}  → 负值=该特征被低估')
    tags_of_interest = ['变速/闪现', '差速', '脑裂', '多面下落', '快交互', '连点爆发', '宽排键',
                        '慢流速', '倒打', '反手', '长连点/交互', '蓝夹黄', '面海', '长条藏键', '判定线干扰',
                        '非线性下落', '扫线', '4k', '5k', '全换', '蓝夹红']
    results = []
    for tag in tags_of_interest:
        got = [i for i in range(n) if row_votes[i] and row_votes[i].get(tag)]
        notgot = [i for i in range(n) if row_votes[i] and not row_votes[i].get(tag)]
        if len(got) < 3:
            continue
        mu_got, mu_not = np.mean(errs[got]), np.mean(errs[notgot])
        results.append((tag, len(got), len(notgot), mu_got, mu_not))
    results.sort(key=lambda r: r[3] - r[4])
    for tag, g, ng, mg, mn in results:
        print(f'{tag:<10} {g:>5} {ng:>5} {mg:>+10.3f} {mn:>+10.3f} {mg-mn:>+8.3f}')

    # ===== 问题2: 特征相关性 =====
    print('\n===== 候选特征 × 投票数 (Spearman) =====')
    feats_arr = [dict(f) for f in feats]
    CAND = {
        '变速/闪现': ['note_speed_non1_ratio', 'note_speed_std', 'note_speed_max', 'note_speed_density',
                     'flash_hold_ratio', 'tempo_change_count', 'speed_volatility'],
        '差速': ['note_speed_std', 'note_speed_max', 'note_speed_non1_ratio'],
        '快交互': ['type_switch_per_sec', 'real_core_notes_per_second', 'pattern_switch_rate',
                  'stair_speed_avg', 'jack_density'],
        '连点爆发': ['short_jack_density', 'long_jack_count', 'jack_density', 'jack_max_run',
                   'real_core_notes_per_second'],
        '脑裂': ['jline_movement_density', 'jline_rotate_density', 'position_entropy',
                'direction_irregularity', 'above_below_cross'],
        '多面下落': ['jline_disappear_density', 'jline_movement_density', 'position_range_used'],
        '面海': ['hold_interference_index', 'avg_hold_duration', 'above_avg_duration_sec'],
        '慢流速': ['speed_volatility', 'tempo_change_count', 'rhythm_entropy'],
        '宽排键': ['position_range_used', 'position_entropy', 'stair_chord_ratio'],
        '反手': ['direction_irregularity', 'above_below_cross', 'jline_rotate_density'],
    }
    for tag, fnames in CAND.items():
        got = [i for i in range(n) if row_votes[i] and row_votes[i].get(tag)]
        if len(got) < 5:
            continue
        cnt = np.array([row_votes[i][tag] for i in got])
        line = f'{tag:<8} n={len(got):>3}: '
        for fn in fnames:
            vals = np.array([feats_arr[i].get(fn, 0) for i in got])
            if vals.std() == 0:
                continue
            rho, p = spearmanr(vals, cnt)
            mark = '**' if p < 0.01 else ('*' if p < 0.05 else '')
            line += f'{fn}={rho:+.2f}{mark} '
        print(line)


if __name__ == '__main__':
    main()
