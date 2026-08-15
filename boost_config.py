# -*- coding: utf-8 -*-
"""v9.0 手动定义的 Boost 特征配置 (5维) — 独立模块, 供 app 与训练脚本共用
核心思路: GB给基线, Boost区分高低定数
无任何压缩函数 — DynamicCap/Sigmoid/Bias 全部移除
"""
# 密度: 2:8权重 (总体真实TPS : 高潮段真实TPS), 只计tap+hold
# 耐力: 高潮段总持续秒数
# 读谱: 权重降低 ~40%
# 高速: 维度已移除 (特征已吸收到配置)
MANUAL_FLAT = [
    # ===== 密度 (2:8 = 总体:高潮段) =====
    ('real_core_notes_per_second', 2.0, 0.08),
    ('above_avg_density_mean',       4.0, 0.352),  # v11.13最终: 密度×1.1
    # 有效单指密度 (同押去冗余, 区分"多指全押顺手"vs"单指连打底力"; volcanic 4押海 vs D321 键盘)
    ('eff_peak_tps_1s',              8.0, 0.18),   # v11.13最终: eff×0.9
    ('eff_avg_tps_1s',               4.0, 0.08),
    # ===== 配置 (全保留) =====
    ('stair_density',                 1.0, 0.022),
    ('stair_speed_avg',               8.0, 0.170),
    ('stair_complexity',              0.2, 0.040),
    ('stair_chord_ratio',             0.3, 0.008),
    ('chord_size_entropy',            0.5, 0.034),
    ('chord_complexity',              0.3, 0.030),
    ('chord_alternation_rate',        0.5, 0.1152), # v11.13最终: 和弦交替×0.6
    ('chord_chord_alt_rate',          0.3, 0.030),
    ('weighted_mf_score_per_sec',    10.0, 0.1272), # v11.13最终: 多押×0.85*0.85≈0.72
    ('discrete_mf_ratio',             0.3, 0.006),
    ('position_entropy',              2.0, 0.054),
    ('avg_chord_size_poly',           2.0, 0.050),
    ('position_range_used',           0.5, 0.078),
    ('trill_density',                 2.0, 0.002),
    ('multi_finger_3plus_events',    10.0, 0.002),
    ('pattern_switch_rate',           1.0, 0.134),
    ('direction_irregularity',        0.5, 0.014),
    ('drag_flick_ratio',              0.2, 0.030),
    # ===== 位移 (修复阈值后: 大位移交互238bpm的每秒位移; 复合=位移×密度) =====
    ('movement_per_second',           6.0, 0.060),  # v11.13最终: 位移×1.0
    ('movement_density_index',       30.0, 0.060),
    # ===== 耐力 (高潮段秒数) =====
    ('above_avg_duration_sec',       30.0, 0.56),   # v11.13最终: 耐力×1.4
    # v12.5: 多面/多线下落维度 (用户: Feeling Blue#47264难点在判定线多面运动, 短长条≈蓝键, hold属性本身不难)
    # 官谱jline位移p95=3.87, FB=5.77/xodus=2.0(单面静态) — 位移活跃区分"多面下落"与"单面多押"
    # v12.5温和版: 0.25→0.15 (权重过大扰动ranked全局校准平衡, MAE从0.404恶化到0.46)
    ('jline_move_disp_per_sec',       2.5, 0.06),  # v12.6: 0.15→0.06 (判定线平移演出虚高: Runengon位移17.98/s贡献0.40但用户未判其难; FB靠窄偏置兜底)
    # v12.5: 全hold域补偿 — 官谱hold_ratio最高0.59(≥0.6的0张), GB学到的hold负关联是官谱巧合;
    # 用户: 短长条≈蓝键, 击打难度等价, 全hold谱不应被压低 (温和版 0.15→0.08)
    ('hold_ratio',                    0.40, 0.08),
    # ===== 读谱 (权重降低 ~40%) =====
    ('tempo_change_count',            50.0, 0.028),
    ('rhythm_entropy',                2.5, 0.058),
    ('type_switch_per_sec',           0.4, 0.080),  # v12.6: 0.100→0.080 (drag参与的类型切换虚高: おぎゃり3.93含大量drag交替)
    ('note_clutter_ratio',            0.05, 0.032),
    ('density_transition_mean',       0.15, 0.026),
    ('density_transition_std',        0.2, 0.0448), # v11.13最终: 密度波动×0.7
    ('hold_interference_index',       0.3, 0.058),
    ('jline_movement_density',        50.0, 0.0228), # v11.13最终: 判定线移动×0.3
    ('jline_rotate_density',          20.0, 0.0307), # v11.13最终: 判定线旋转×0.64
    ('jline_disappear_density',       20.0, 0.046),
    ('speed_volatility',              0.1, 0.050),
    ('above_below_cross',             0.3, 0.044),
    # ===== 重键 jack (特征已计算但此前未进入boost; 保守权重, 防误伤同线谱) =====
    ('jack_density',                  2.0, 0.030),
    ('jack_max_run',                  2.0, 0.021),  # v11.13最终: 纵连×0.7
    ('same_line_jack_ratio',          0.1, 0.050),
    ('long_jack_count',               4.0, 0.020),
    # v12.3回滚: global_jack/24th/16th boost权重试验过冲 (cap抹平60137与xodus的24分差异, 三锚全抬);
    # 双指/多指差异改由app.py类型偏置处理
    # ===== 差速/闪现 (音符级 speed 与 visibleTime; 阈值提高+权重降低, 仅极端差速触发) =====
    # v12.2: 权重减半 — 变速演出类特征社区共识定价保守 (xodus#294全程高速演出note_speed_non1=0.95
    # 较Apollo的0.01高130倍但真实难度相当, 原权重贡献+0.42虚高)
    # v12.4: 再减半 (xodus演出虚高仍未压够)
    ('note_speed_non1_ratio',         0.20, 0.015),
    ('note_speed_std',                1.5, 0.015),
    ('note_speed_max',                4.0, 0.005),
    ('note_speed_density',            4.0, 0.008),
    ('flash_hold_ratio',              0.1, 0.120),
    # ===== 和弦重键 (chord jack: 同线连续和弦快速重复) =====
    # density 降权 (愚人节多押谱误伤); 3plus_pairs(重键4k) 保留
    ('chord_jack_density',            2.0, 0.080),
    ('chord_jack_3plus_pairs',        4.0, 0.040),
    # ===== v11.5 极端配置 (AP难度视角: 出现即拉高; 阈值=官谱IN/AT段p95级; 权重减半: GB已吸收信息, boost仅补充) =====
    ('thirtysecond_run_max',          40.0, 0.030),
    ('thirtysecond_run_ratio',        0.40, 0.050),
    ('cross_hand_density',            4.0, 0.050),
    ('lane_switch_density',           5.0, 0.015),
    ('jline_relative_cross',          0.10, 0.040),
    # ===== v11.7 纯drag滑动谱 (Feeling Blue类: drag_ratio高+drag密度高; RPE drag带holdTime是格式特性非难度) =====
    ('drag_per_sec',                  3.0, 0.040),  # v11.13最终: drag×0.4
]
