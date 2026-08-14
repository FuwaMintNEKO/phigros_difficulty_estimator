# 难配置完整清单 (AP难度视角) — v11.5 特征覆盖审计

> 来源: ① 项目文档对话记录 ② phira/kyou站标签 ③ B站术语 ④ Phigros社区通用
> 目标: 官谱按AP(All Perfect)难度定价 → 极端配置出现即拉高定数; 清单对照现有特征, 标记缺失

## 一、击打配置类 (手部操作难度)

| # | 配置 | 定义/社区说法 | 现有特征 | 状态 |
|---|---|---|---|---|
| 1 | **交互 Alternating** | 左右手交替击打; 16/24/32/48分 | short_interval_ratio, thirtysecond_run_max/ratio, interaction_ms_run_max(新) | ✅ 拍域+毫秒域 |
| 2 | **楼梯 Stair** | 阶梯状连续击打; 快速楼梯/长楼梯/48分楼梯 | stair_density, stair_speed_avg, stair_complexity, stair_purity, stair_chord_ratio | ✅ |
| 3 | **纵连 Jack** | 同键快速连打; 短纵连(3-4)/长纵连(5+) | jack_density, jack_max_run, short_jack_count, long_jack_count, same_line_jack_ratio | ✅ |
| 4 | **叠键/重键 Chord Jack** | 同位置连续和弦快速重复 ("叠键地狱" B站) | chord_jack_density, chord_jack_3plus_pairs | ✅ |
| 5 | **多押 Chord** | 同时击打2-6键; 越多押越难 (用户原话) | avg_chord_size, avg_chord_size_poly, chord_entropy_norm, multi_finger_3plus_events, multi_line_sim_events | ✅ |
| 6 | **换手 Hand Cross** | 双手交叉击打 (用户明确提到) | cross_hand_density(新boost), cross_hand_event_count, jline_relative_cross(新) | ✅ 新增 |
| 7 | **出张 Stretch** | 手指跨屏/远距离伸张击打 ("劲爆出张" B站) | movement_per_second, max_movement, lane_switch_density(新) | ✅ 新增 |
| 8 | **反手** | 非惯用手主导段 | — | ❌ 难量化 |
| 9 | **双押交互/和弦流** | 双手同时交互 | chord_alternation_rate, chord_chord_alt_rate | ✅ |
| 10 | **转圈/环绕** | 音符环绕旋转配置 | position_entropy 间接 | ⚠️ 无显式特征 |
| 11 | **拇指谱/单手** | 单指/拇指操作 (B站"拇指、双食指篇") | eff_avg_tps_1s 间接 | ⚠️ 无显式特征 |
| 12 | **交互+长条组合** | 长条按住时交互 | hold_interference_index | ⚠️ 间接 |

## 二、音符速度类 (用户3316行原话: 100bpm的32分=200bpm的16分)

| # | 配置 | 定义 | 现有特征 | 状态 |
|---|---|---|---|---|
| 13 | **高速音符群** | 24/32/48分音符群; 贡献按真实速度(BPM归一) | fast_note_density_16th(拍域), fast_ms_050/100/150_ratio(新,毫秒域) | ✅ 新增毫秒域 |
| 14 | **爆发段 Burst** | 短时高密度爆发/尾杀 | tap_burst_05_top5, burst_intensity_mean, miniburst_density, max_consecutive_burst | ✅ |
| 15 | **底力/耐力** | 全程高密度持续 (phira标签"底力") | above_avg_duration_sec, stamina_high_sec | ✅ |
| 16 | **变速欺诈** | 流速突变/变速谱 (HAL 30000案例) | tempo_change_count, speed_volatility, tempo_change_log_density(新) | ✅ 新增log |

## 三、读谱/视觉类 (Phigros特有)

| # | 配置 | 定义 | 现有特征 | 状态 |
|---|---|---|---|---|
| 17 | **判定线移动** | 判定线位移/旋转/消失 | jline_movement_density, jline_rotate_density, jline_disappear_density | ✅ |
| 18 | **判定线表演** | 表演段按键退出屏幕外复用 | jline_movement_density 间接 | ⚠️ |
| 19 | **音符闪现** | visibleTime 极短 | flash_note_ratio, flash_hold_ratio | ✅ |
| 20 | **差速音符** | 音符speed≠1 | note_speed_non1_ratio, note_speed_std, fast_hold_count | ✅ |
| 21 | **长条干扰** | 长条视觉遮挡 | hold_interference_index (v8.4修复后0.0452) | ✅ |
| 22 | **故事板/表演谱** | 大量视觉事件 (phira标签"故事板"74) | jline_* 间接 | ⚠️ 无显式 |
| 23 | **骗手/误导** | 反直觉排列 | — | ❌ 难量化 |
| 24 | **急停 Stop-Go** | 突然停顿再继续 | stop_go_count(GB排除中!) | ⚠️ 被GB排除 |

## 四、轨道/定轨类

| # | 配置 | 定义 | 现有特征 | 状态 |
|---|---|---|---|---|
| 25 | **定轨4k/5k/6k** | 固定轨道密集击打 (Chart_SP案例, 双指无解) | tracks_4plus/5plus/6plus_sec + app.py加成0.15/0.55/1.0 | ✅ |
| 26 | **键盘段** | phira标签"键盘"90 | 同25 | ✅ |
| 27 | **跨线配置** | 判定线不动但音符跨线穿梭 (3rd Avenue案例) | jline_relative_cross(新), lane_switch_density(新), crossline_chain_max(新) | ✅ 新增 |

## 五、谱面结构类

| # | 配置 | 定义 | 现有特征 | 状态 |
|---|---|---|---|---|
| 28 | **尾杀** | 结尾爆发 (DistortedFate案例) | tail_core_share, tail窗口特征 | ✅ |
| 29 | **可馅蜜/多面型** | 多押可拆连打通过 (user明确: 多面型可馅蜜协调) | multi_line_sim_events, ML_HEAVY条件缩放 | ✅ |
| 30 | **hold-heavy** | 长条主导 (Feeling Blue -6.62案例) | avg_hold_duration, hold_interference_index | ⚠️ 仍低估, 待hold语义特征 |

## 已实施的 v11.5 新增特征 (13个)
- 变速: tempo_change_log_density, speed_event_log_density, speed_volatility_log
- 跨线: lane_switch_count/ratio/density, crossline_chain_max/ratio, jline_relative_cross
- 32分交互: thirtysecond_run_max/ratio/count
- 速度归一化: fast_ms_050/100/150_ratio, interaction_ms_run_max/ratio
- boost权重(减半后): thirtysecond_run_max 0.03, thirtysecond_run_ratio 0.05, cross_hand_density 0.05, lane_switch_density 0.015, jline_relative_cross 0.04
- app.py: EXTREME_FEATS_COND 缩放 (双指×1.3拉高, 多指×0.85压低, 仅自制谱)

## 待办 (难量化, 需讨论)
- 反手/骗手/转圈: 需谱面设计语义, 暂无法可靠量化
- hold语义: Feeling Blue 仍-5.9, hold-heavy谱低估是最大方差源 (下一个方向)
- 故事板: 表演事件数量可作为"视觉干扰"代理特征 (phira故事板标签可验证)
- stop_go: 从GB排除列表恢复或加boost权重
