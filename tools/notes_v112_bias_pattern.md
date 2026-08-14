# v11.2 t1: 上架589谱偏差Top分析与特征模式定位

> 实验: tools/exp_v112_bias_top.py ｜ 数据: data/phira/feats_cache_v11.pkl（上架589张, diff>10）
> 模型: models/6dim_model_v11_1.pkl（GB+条件boost+校准, **不含定轨加成**）
> 口径: pred - diff（社区定数=第二权威）；官谱同段 P75/P90 按 diff 段（<13/13-14/14-15/15-16/16-17/>=17）计算
> 详细数据: logs/exp_v112_bias_top.txt

## 一、总体偏差（589张）

| 段 | n | bias均值 | MAE | RMSE |
|---|---|---|---|---|
| 13-14 | 55 | -0.028 | 0.714 | 0.945 |
| 14-15 | 163 | **+0.121** | 0.477 | 0.601 |
| 15-16 | 240 | **+0.094** | 0.543 | 0.697 |
| 16-17 | 99 | +0.013 | 0.437 | 0.631 |
| >=17 | 17 | **-0.307** | 0.483 | 0.559 |
| ALL | 589 | +0.058 | 0.538 | 0.717 |

16+ 分组：多指(mf3>=30) **-0.125** ｜ 双指(mf3<=5) -0.027 ｜ 混合 +0.099。

要点：
- 14-16 段整体高估约 +0.09~+0.12（v11 校准后仍残余）；>=17 段低估 -0.31（社区 17+ 定数虚高，外推段有意压低）。
- 16+ 多指谱已基本压平（-0.125），**双指谱不再低估**（v11.1 抬 eff 生效，-0.027）。

## 二、高估 Top20 特征模式（bias 最大，+2.16 ~ +1.24）

代表谱：Ocean Blue(+2.16, diff11.1)、Requiem Cataclysm(+1.67)、cyanine(+1.67)、IMPACT(+1.65)、Saintelmo(+1.62)、Plumes(+1.57)、Doping Dance(+1.51) 等。

### 特征系统性超官谱同段范围（n=20）

| 特征 | 超P90 | 超P75 | 语义 |
|---|---|---|---|
| tempo_change_count | **13** | 3 | 变速/读谱量超官谱 |
| real_core_notes_per_second | **11** | 6 | 高潮段真实TPS超官谱 |
| weighted_mf_score_per_sec | **11** | 6 | 多押加权分数/秒超官谱 |
| total_notes | **11** | 6 | 总音符数超官谱 |
| movement_per_second | 9 | 10 | 判定线移动量超官谱 |
| above_avg_density_mean | 8 | 8 | 高潮段密度超官谱 |
| eff_avg_tps_1s | 8 | 4 | 有效单指密度超官谱 |
| eff_peak_tps_1s | 7 | 3 | 有效峰值密度超官谱 |
| speed_volatility | 7 | 1 | 变速波动超官谱 |
| multi_line_sim_events | 5 | 2 | 多线同押超官谱 |
| stair_density | 5 | 4 | 楼梯密度超官谱 |

### 核心结论：高估谱 = "双指堆料谱"

1. **18/20 是双指谱（mf3=0）**，仅 2 张多指（Bl∞min' mf3=34 dens=15.7、火狐之舞 mf3=62 dens=17.1，均为高密度真材实料）。→ 多指堆料已被 v11 条件 boost 压住，**当前高估主源不在多指**。
2. 高估谱的密度/多押/读谱量特征全面落在官谱 P75-P90 之上甚至超 P90：中位数 real_core_notes_per_second 6.84 vs 官谱15-16段 P75 6.34、weighted_mf_score_per_sec 11.34 vs P75 10.02、movement_per_second 27.81 vs P75 25.78、tempo_change_count 676 vs P75 667。
3. **双指谱 eff 抬升 1.5 的副作用**：v11.1 为修正官谱双指高难低估（-0.23）而抬 eff，但上架谱中大量"双指密度堆料谱"（社区定数并不高，如 Requiem Cataclysm 社区 15.0 但 dens 12.07/eff_avg 7.91 已达官谱 16-17 段水平）同样享受 1.5 倍 eff 加成 → 高估被放大。这是一个**双指谱内部分化**问题：官谱双指高难是"高 eff + 低密度"，堆料谱是"高 eff + 高密度"，后者被抬过头。
4. 低估侧无此模式：高估 Top20 中特征低于官谱 P25 的极少（stair_speed_avg 4、speed_volatility 6、duration_sec 4），**高估完全由"特征超官谱域"驱动，而非官谱域内噪声**。

## 三、低估 Top20 特征模式（bias 最小，-3.06 ~ -1.53）

代表谱：甜甜饕餮糖糖(-3.06, diff13)、寄明月(-2.70)、Feeling Blue(-2.67, diff16.7)、魔理沙(-2.56, diff16.5)、时落之雨(-2.44)、After Rain(-2.38) 等。

### 特征系统性低于官谱同段范围（n=20）

| 特征 | 低P25 | 低P10 | 语义 |
|---|---|---|---|
| eff_avg_tps_1s | **16** | 14 | 有效单指密度全面偏低 |
| above_avg_density_mean | **15** | 13 | 高潮段密度偏低 |
| real_core_notes_per_second | **15** | 10 | 高潮段真实TPS偏低 |
| movement_per_second | **15** | 12 | 移动量偏低 |
| eff_peak_tps_1s | 14 | 13 | 有效峰值偏低 |
| stair_density | 14 | 9 | 楼梯密度偏低 |
| total_notes | 13 | 10 | 总音符数偏低 |
| tempo_change_count | 9 | 6 | 变速量偏低 |

### 核心结论：低估谱 = "特征不足谱"，分三个子模式

1. **低密度普通谱被社区高定**（约 14/20）：dens/eff/movement/total_notes 全面低于官谱 P25（甚至 P10），如甜甜饕餮糖糖 dens=4.5/eff_avg=3.2（官谱 13-14 段 P25 为 5.81/4.35）、3rd Avenue dens=5.97/eff_avg=3.80 却标 15.0。模型按官谱标尺"说实话"给了低分 → 这类谱社区定数虚高是低估主因之一，模型未必错。
2. **纯 hold 谱系统性低估**：Feeling Blue 全部音符为 hold（hold_ratio=1.00, n=820, dens=11.3, mf3=43）社区 16.7，模型 14.03。特征体系对"全 hold 高难"（连打长条、hold 段读谱）无刻画 → 需要 hold 类特征（hold 密度/hold 变速）。
3. **超长 drag 谱系统性低估**：魔理沙 4001 音符/234s/70% drag（nps 17.1 但 drag 不产生有效单指密度）社区 16.5，模型 13.94；时落之雨 68% drag 社区 12.2 模型 9.76。drag 在现有体系权重低，长 drag 铺底谱被整体压低。

注：低估 Top20 中 mf3 超 P90 的 5 张、stair_speed_avg 超 P90 的 5 张、duration_sec 超 P90 的 6 张属于"个别特征极端但整体特征不足"，与主流模式不同源。

## 四、特征级诊断（全体589 vs 高估20 vs 低估20 vs 官谱>=14 中位数）

| 特征 | 589 | 高估20 | 低估20 | 官谱>=14 |
|---|---|---|---|---|
| above_avg_density_mean | 8.70 | 9.00 | **4.94** | 8.83 |
| real_core_notes_per_second | 5.77 | **6.84** | **3.01** | 5.60 |
| eff_avg_tps_1s | 6.10 | 6.39 | **3.77** | 6.34 |
| eff_peak_tps_1s | 12.0 | 13.0 | **6.0** | 13.0 |
| weighted_mf_score_per_sec | 8.35 | **11.34** | **3.36** | 7.22 |
| movement_per_second | 21.16 | **27.81** | **7.17** | 20.89 |
| stair_density | 2.91 | 3.17 | **0.50** | 2.87 |
| tempo_change_count | 596 | **676** | 321 | 545 |
| total_notes | 1063 | 1150 | **661** | 1000 |
| speed_volatility | 165 | 258 | 302 | 332 |

- 高估谱与低估谱在密度系特征（above_avg/real_core/eff_avg/eff_peak/movement/stair/total_notes）上呈**镜像偏移**：高估谱超官谱 P75/P90，低估谱低于官谱 P25/P10。
- speed_volatility 存在极端 OOD 值（Saintelmo 36 万、Doping Dance 64 万、Fine Logic 383 万、颜 2178 万 vs 官谱 P90≈3223），但方向性不强（最大值的 8 张中高估/低估混杂），属**特征域偏移信号而非偏差主因**。

## 五、对 v112 特征修正的建议方向（供 fe-engineer）

1. **双指密度堆料 vs 真双指耐力分化**：当前条件 boost 按 mf3 分组，双指谱（mf3<=5）一律抬 eff 1.5。建议引入密度条件：高密度双指谱（dens 超官谱 P90 的"堆料型"）不抬或少抬 eff，低密度双指耐力谱保持抬升。→ 与 t2 的 eff_density_ratio 方案联动。
2. **hold 高难刻画缺失**：全 hold 谱（Feeling Blue）偏差 -2.67。建议新增 hold 类特征（hold 段密度、hold 变速、hold 同押）或在现有特征中提高 hold 贡献权重。
3. **drag 长铺底谱低估**：drag_ratio>0.5 的长谱（魔理沙）偏差 -2.56。drag 特征权重偏低，建议评估 drag 类特征对定数的相关性后决定是否修正（注意官谱 drag 谱也存在，需保证官谱不失真）。
4. **tempo_change_count / movement_per_second 超域是读谱量超标的通用信号**：高估 Top20 中 13/20、9/20 超官谱 P90，可作为"读谱复杂度虚高"的判定特征，供特征公式修正（非加 cap）。
5. **speed_volatility 极端值**：存在 10^6~10^7 量级 OOD 值，建议检查特征计算（波动率平方/未归一化）是否有异常放大，必要时做域裁剪（对数/分位变换），这是特征域问题而非偏差直接主因。

---
*数据来源: logs/exp_v112_bias_top.txt（tools/exp_v112_bias_top.py 运行结果）*
