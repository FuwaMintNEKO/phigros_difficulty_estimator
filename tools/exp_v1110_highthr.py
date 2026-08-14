# -*- coding: utf-8 -*-
"""高难段判定阈值搜索: 用特征区分 16.5+ 与 16-16.5, 统计误分率"""
import os, sys, numpy as np, io, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, _ROOT)
with open(os.path.join(_ROOT, 'data', 'phira', 'feats_cache_v11.pkl'), 'rb') as f:
    cache = pickle.load(f)
ranked = [r for r in cache['ranked'] if r['diff'] and r['diff'] > 10]
ds = np.array([round(r['diff'],1) for r in ranked])
A = [(i,r) for i,r in enumerate(ranked) if 16 <= ds[i] < 16.5]
B = [(i,r) for i,r in enumerate(ranked) if ds[i] >= 16.5]
def g(r, k): return r['feats'].get(k, 0)

# 候选判定: 组合特征, 找在B中命中率高、A中误判低的阈值
conds = {
  'mf3>=25': lambda r: g(r,'multi_finger_3plus_events') >= 25,
  'mf3>=30': lambda r: g(r,'multi_finger_3plus_events') >= 30,
  'nps>=11': lambda r: g(r,'real_notes_per_second') >= 11,
  'nps>=11.5': lambda r: g(r,'real_notes_per_second') >= 11.5,
  'jack>=400': lambda r: g(r,'global_jack_count') >= 400,
  'jack>=500': lambda r: g(r,'global_jack_count') >= 500,
  'dens>=10.5': lambda r: g(r,'above_avg_density_mean') >= 10.5,
  'hold>=200': lambda r: g(r,'hold_count') >= 200,
  'tracks_active>=150': lambda r: g(r,'tracks_active_sec') >= 150,
}
print(f'{"判定":<22}{"B命中":>8}{"A误判":>8}')
for name, fn in conds.items():
    b_hit = sum(1 for _, r in B if fn(r)); a_hit = sum(1 for _, r in A if fn(r))
    print(f'{name:<22}{b_hit/len(B)*100:>7.0f}%{a_hit/len(A)*100:>7.0f}%')
# 组合: 高难= (mf3>=25 且 nps>=10.5) 或 jack>=450
combos = {
  '(mf3>=25&nps>=10.5)|jack>=450': lambda r: (g(r,'multi_finger_3plus_events')>=25 and g(r,'real_notes_per_second')>=10.5) or g(r,'global_jack_count')>=450,
  '(mf3>=20&nps>=11)|jack>=400': lambda r: (g(r,'multi_finger_3plus_events')>=20 and g(r,'real_notes_per_second')>=11) or g(r,'global_jack_count')>=400,
  '(mf3>=30)|(nps>=12)': lambda r: g(r,'multi_finger_3plus_events')>=30 or g(r,'real_notes_per_second')>=12,
  '(mf3>=20&dens>=10)|jack>=350': lambda r: (g(r,'multi_finger_3plus_events')>=20 and g(r,'above_avg_density_mean')>=10) or g(r,'global_jack_count')>=350,
  'effpeak>=16&mf3>=15': lambda r: g(r,'eff_peak_tps_1s')>=16 and g(r,'multi_finger_3plus_events')>=15,
  '(mf3>=25)|(jack>=450)|(nps>=12)': lambda r: g(r,'multi_finger_3plus_events')>=25 or g(r,'global_jack_count')>=450 or g(r,'real_notes_per_second')>=12,
}
print()
for name, fn in combos.items():
    b_hit = sum(1 for _, r in B if fn(r)); a_hit = sum(1 for _, r in A if fn(r))
    print(f'{name:<40}{b_hit/len(B)*100:>7.0f}%{a_hit/len(A)*100:>7.0f}%')
# 看B中未被命中的谱 (用最后一个组合)
fn = combos['(mf3>=25)|(jack>=450)|(nps>=12)']
miss = [(ds[i], r['name'][:24], round(g(r,'multi_finger_3plus_events'),0), round(g(r,'global_jack_count'),0), round(g(r,'real_notes_per_second'),1)) for i,r in B if not fn(r)]
print('\nB中未命中(16.5+却不像高难):')
for m in sorted(miss, reverse=True): print('  ', m)
fake = [(ds[i], r['name'][:24], round(g(r,'multi_finger_3plus_events'),0), round(g(r,'global_jack_count'),0), round(g(r,'real_notes_per_second'),1)) for i,r in A if fn(r)]
print('\nA中误判(16-16.5却被判高难):')
for m in sorted(fake, reverse=True): print('  ', m)
print('DONE')