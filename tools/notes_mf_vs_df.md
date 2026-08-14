# t2: 16+ 段多指谱 vs 双指谱特征分离度分析

> 实验脚本: tools/exp_mf_vs_df_feats.py ｜ 原始输出: logs/exp_mf_vs_df_feats.txt

## 1. 实验设计

- **数据**: data/phira/json/*.json 全部 615 张上架谱，取社区定数 **diff>=16.0** 的 **116 张** 16+ 段谱面
- **分组**: 按 `multi_finger_3plus_events`(mf3) —— **多指组 mf3>=30 (33 张)** / **双指组 mf3<=5 (64 张)** / 中间带 19 张
- **分离度**: `|均值差| / 合并标准差(pooled std, ddof=1)`，越大越能区分两组
- 特征由 `feature_extractor.extract_features` 提取（与生产管线同源）

## 2. 关键结果

### 2.1 分离度排序（多指组 vs 双指组）

| 特征 | 多指均值 | 双指均值 | 分离度 | 方向 |
|---|---|---|---|---|
| **above_avg_density_mean** | 13.36 | 10.41 | **1.634** | 多指> |
| **real_core_notes_per_second** | 8.66 | 6.95 | **1.211** | 多指> |
| above_avg_duration_sec | 925.5 | 699.4 | 0.812 | 多指> |
| weighted_mf_score_per_sec | 13.60 | 10.25 | 0.807 | 多指> |
| chord_alternation_rate | 2.49 | 1.98 | 0.775 | 多指> |
| tap_burst_top5 | 1.16 | 0.99 | 0.740 | 多指> |
| eff_peak_tps_1s | 17.67 | 15.02 | 0.699 | 多指> |
| fast_note_density_32nd | 0.164 | 0.033 | 0.669 | 多指> |
| movement_per_second | 31.12 | 27.75 | 0.421 | 多指> |
| eff_avg_tps_1s | 7.92 | 7.45 | 0.362 | 多指> |
| notes_per_second | 10.32 | 9.73 | 0.275 | 多指> |
| stair_speed_avg | 14.00 | 14.88 | 0.223 | 多指< |
| jline_movement_density | 45.9 | 49.5 | 0.059 | — |
| type_switch_per_sec | 1.03 | 1.02 | 0.026 | — |

（`multi_finger_3plus_ratio` 分离度 4.71 是分组定义本身，不具判别意义；`multi_finger_density` 1.444 可参考）

### 2.2 特征与社区定数相关性（16+ 全体 116 张）

| 特征 | Pearson | Spearman |
|---|---|---|
| above_avg_density_mean | +0.537 | +0.427 |
| above_avg_duration_sec | +0.568 | +0.399 |
| eff_avg_tps_1s | +0.419 | +0.386 |
| real_core_notes_per_second | +0.444 | +0.332 |
| eff_peak_tps_1s | +0.439 | +0.314 |
| movement_per_second | +0.335 | +0.293 |
| chord_alternation_rate | +0.282 | +0.289 |
| weighted_mf_score_per_sec | +0.282 | +0.203 |
| type_switch_per_sec | +0.227 | +0.197 |
| fast_note_density_32nd | +0.189 | +0.166 |
| tap_burst_top5 | +0.138 | +0.091 |
| stair_speed_avg | +0.001 | +0.045 |
| jline_movement_density | -0.093 | -0.101 |

## 3. 核心结论

1. **最能区分多指/双指的特征**（分离度 Top3）:
   - `above_avg_density_mean`（高密度段平均密度，sep=1.634）—— 多指谱高密度段密度显著更高
   - `real_core_notes_per_second`（核心音符真实密度，sep=1.211）—— 多指谱核心击打密度更高
   - `above_avg_duration_sec`（高密度段持续时长，sep=0.812）—— 多指谱高密度段持续更久

2. **"判别力 × 难度相关"双高特征**（v11 修正的最佳候选依据）:
   - **`above_avg_density_mean`**: 分离度 1.634 + 与定数 Pearson 0.537 / Spearman 0.427 —— 既是难度指示器又是多指判别器
   - **`real_core_notes_per_second`**: 分离度 1.211 + Pearson 0.444 / Spearman 0.332 —— 同上
   - 这两特征在模型里会同时"推高"多指谱得分——多指谱密度天然高 → 模型给高分 → **与"社区多指虚高应压低"的目标方向相反**，是 16+ 段系统性偏差的特征层根源

3. **双指组特征画像**: 双指谱在 `above_avg_density_mean`(10.41 vs 13.36)、`real_core_notes_per_second`(6.95 vs 8.66)、`eff_peak_tps_1s`(15.02 vs 17.67)、`fast_note_density_32nd`(0.033 vs 0.164) 全面显著低于多指组——即双指谱靠"持久高密度"而非"瞬时峰值/多指配置"拿难度，社区对这类"纯双指耐力谱"定数偏低（-0.15），模型也相应偏低，需**抬高**。

4. **外推段与多指的强耦合**: 多指组社区定数 16.86±0.76（上限 18.5），双指组 16.29±0.27（**上限仅 17.1**）——社区定数 17.7+ 的外推段谱**全部是多指谱**，双指谱不存在 17.7+ 样本。这解释了为何外推段（t4）偏差全部集中在多指谱上。

5. **对 v11 改良的含义**:
   - 若做"多指/双指条件化"修正，**`above_avg_density_mean` 与 `real_core_notes_per_second` 是天然的分组判别特征**（无需依赖 mf3 本身），可在模型输出后按这两特征分段施加方向相反的 delta（多指 -，双指 +）
   - `type_switch_per_sec` / `jline_movement_density` / `stair_speed_avg` 分离度极低（<0.25），不适合做判别特征
   - 双指组内特征与定数相关普遍低于多指组（如 weighted_mf 多指组 r=0.25 vs 双指组 0.005），说明双指谱难度主要由其他维度（耐力/尾杀）决定，需 t3 的尾杀窗口特征补充

## 4. 复现

```bash
C:\Python314\python.exe tools\exp_mf_vs_df_feats.py
# 输出: logs/exp_mf_vs_df_feats.txt
```
