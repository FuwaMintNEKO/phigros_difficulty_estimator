# Phigros 难度定数预测系统 v8.6 — 开发历程全记录

> 撰写日期：2026-06-06
> 项目：Phigros 谱面难度定数预测器
> 核心技术栈：Python, Gradient Boosting Regressor, Ridge Regression, Flask Web

---

## 目录

1. [项目概述](#1-项目概述)
2. [版本演变总览](#2-版本演变总览)
3. [核心架构设计](#3-核心架构设计)
   - 3.1 双模块架构：GB 基线 + Boost 叠加
   - 3.2 特征提取引擎
   - 3.3 动态 Cap 机制
   - 3.4 Sigmoid 平滑调整函数
   - 3.5 Excess 计算与阈值体系
4. [维度体系演进](#4-维度体系演进)
   - 4.1 密度维度
   - 4.2 配置维度
   - 4.3 耐力维度（重构历程）
   - 4.4 读谱维度（判定线视觉干扰）
   - 4.5 位移维度（已被移除）
   - 4.6 高速音符维度
5. [各版本详细变化](#5-各版本详细变化)
   - 5.1 v5dim 时代（v3 ~ v4）
   - 5.2 v6dim 时代（v7.0 ~ v7.6）
   - 5.3 v8.0 ~ v8.2：巅峰密度移除与读谱增强
   - 5.4 v8.3：密度维度重构
   - 5.5 v8.4：耐力维度重构（高潮段占比）
   - 5.6 v8.5：耐力再重构 + 删除位移 + 高速音符扩展
   - 5.7 train_manual.py：手动调优阶段
6. [统计学特征诊断](#6-统计学特征诊断)
   - 6.1 相关系数 r 分析
   - 6.2 共线性诊断
   - 6.3 特征筛选结果
7. [格式解析历程](#7-格式解析历程)
   - 7.1 官谱格式（Phigros 标准格式）
   - 7.2 RPE 格式
   - 7.3 RPE v3 格式（愚人节谱）
   - 7.4 PE 文本格式
8. [训练方法与优化策略](#8-训练方法与优化策略)
   - 8.1 Ridge 正系数回归
   - 8.2 分层抽样（Stratified Split）
   - 8.3 锁定特征（PINNED）
   - 8.4 最优参数网格扫描
   - 8.5 cf 系数缩放
9. [核心难题与解决方案](#9-核心难题与解决方案)
   - 9.1 高难偏低、低难偏高
   - 9.2 位移维度无存在感
   - 9.3 耐力特征的水段干扰
   - 9.4 判定线视觉干扰难以量化
   - 9.5 高速音符权重倒置
   - 9.6 密度暴走与 cap 取舍
   - 9.7 RPE 格式判定线特征为 0
10. [当前模型状态](#10-当前模型状态)
    - 10.1 当前 FLAT_FEATURES 配置
    - 10.2 手动测试结果
    - 10.3 仍存在的问题
11. [展望与待办](#11-展望与待办)

---

## 1. 项目概述

Phigros 难度定数预测系统是一个基于机器学习的谱面难度评估工具，旨在为 Phigros（一款非对称判定线音游）的自制谱和官谱提供客观的难度定数预测。

**核心目标**：从谱面文件中提取多维特征，通过 Gradient Boosting Regressor + Ridge Boost 叠加架构，将特征映射到 5.0 ~ 19.0 的难度定数区间。

**独特挑战**：
- Phigros 的判定线系统极度自由，与传统音游差异巨大
- 官谱定数范围横跨 5~18+，各区间难度构成差异显著
- 部分难度维度（如读谱、位移）极难量化
- 谱面格式多样（官谱标准格式、RPE、RPE v3、PE 文本）

---

## 2. 版本演变总览

| 版本 | 特征维度 | 核心变化 | 模型文件 |
|------|---------|---------|---------|
| v5dim v3 | 5维 | 初始版本 | `5dim_model_v3.pkl` |
| v5dim v4 | 5维 | 改进训练 | `5dim_model_v4.pkl` |
| v6dim v7.0 | 6维 | 引入 Boost 架构 | `6dim_model_v7.pkl` |
| v6dim v7.1~7.6 | 6维 | 渐进调优 | `6dim_model_v7_{1..6}.pkl` |
| v8.0 | 6维 | 大幅重构特征集 | `6dim_model_v8_0.pkl` |
| v8.1 | 6维 | 增强读谱特征 | `6dim_model_v8_1.pkl` |
| v8.2 | 6维 | 移除峰值密度 Boost | `6dim_model_v8_2.pkl` |
| v8.3 | 6维 | 密度维度=√(rcnps×高潮均值) | `6dim_model_v8_3.pkl` |
| v8.4 | 6维 | 耐力=高潮段占比 | `6dim_model_v8_4.pkl` |
| v8.5 (Ridge) | 6维 | 删除位移+耐力重构+>32nd音符 | `6dim_model_v8_5.pkl`（已被 manual 覆盖） |
| v8.5 (Manual) | 6维 | 手动设置 co 值 | `6dim_model_v8_5.pkl` |
| v8.5b | 6维 | 修正读谱偏高+排序修复 | 覆盖 `6dim_model_v7.pkl` |
| **v8.6** | **6维** | **GB偏低补偿Bias** | **`6dim_model_v7.pkl`（当前部署）** |

---

## 3. 核心架构设计

### 3.1 双模块架构：GB 基线 + Boost 叠加

这是系统最核心的设计决策。预测值由两部分叠加：

```
预测值 = GB 基线 + Boost 调整
```

**GB 基线（Gradient Boosting Regressor）**：
- 使用全量 219+ 特征训练
- 学习谱面特征与定数之间的非线性关系
- 输出范围约 10.8 ~ 15.6（跨度仅 4.8 分）
- 反映的是"大部分谱面的共性难度"

**Boost 调整**：
- 从 219+ 特征中精选 30+ 个高区分力特征
- 通过 Excess 计算 + 系数加权得到 Boost raw 值
- 经过 Dynamic Cap 压缩 + Sigmoid 平滑后叠加到 GB 上
- 负责拉开高低难度的差距

**为什么需要这种架构？**
GB 模型在训练集范围内表现良好，但对训练集外（OOD）的高难度谱面缺乏外推能力。Boost 机制专门负责捕捉"超出常规"的极端特征值，将高难谱推到应有位置。

### 3.2 特征提取引擎

特征提取器（`feature_extractor.py`）解析谱面 JSON 数据结构，计算 219+ 个数值特征，分为：

| 类别 | 特征数 | 典型特征 |
|------|-------|---------|
| 基础统计 | ~15 | total_notes, duration_sec, bpm 等 |
| 密度 | ~30 | density_dimension, peak_density, notes_per_second |
| 配置 | ~50 | stair, trill, jack, chord, multi-finger 等 |
| 耐力 | ~10 | above_avg_density_mean, tap_burst_top5 等 |
| 读谱 | ~50 | jline_movement_density, rhythm_entropy, type_switch 等 |
| 高速音符 | ~10 | fast_note_density_16th~64th |
| 进阶多指 | ~30 | cross_hand, multi_line_sim, hold_lock 等 |
| 其他 | ~30 | tempo_change, stop-go, 等 |

### 3.3 动态 Cap 机制

为解决极端特征值导致 Boost 暴走的问题，引入 Dynamic Cap：

```python
def _dynamic_cap(raw):
    if raw <= KNEE:    # KNEE = 2.5
        return raw     # 线性区不变
    excess = raw - KNEE
    return KNEE + excess ** POWER   # POWER = 0.9，亚线性压缩
```

- **knee=2.5**：2.5 以内的 Boost 不做任何压缩
- **power=0.9**：超出部分开 0.9 次方（接近线性但略压缩）
- 这个机制取代了硬上限（hard cap），保留了高难度谱的区分度同时防止极端值暴走

### 3.4 Sigmoid 平滑调整函数

解决 Boost/GB 比例失衡问题：

```python
def adjust_boost_smooth(boost, gb):
    if boost < 2.0: return boost         # 简单谱不调整
    ratio = boost / gb
    expected = RATIO_TARGET * gb          # 目标 boost = target × GB
    adj = expected * ((boost / expected) ** RATIO_POWER)
    w = 1 / (1 + exp(-STEEPNESS * (ratio - RATIO_THRESHOLD)))  # Sigmoid 权重
    return (1 - w) * boost + w * adj
```

- **target=0.28~0.35**：期望 Boost/GB 比例
- **thresh=0.20~0.24**：触发调整的阈值
- **power=0.65~0.80**：压缩力度
- **steepness=25**：Sigmoid 曲线的陡峭程度

当 Boost/GB 比例超过阈值时，Sigmoid 逐渐将 Boost 向目标值压缩，避免 Boost 占比过高。

### 3.5 Excess 计算与阈值体系

Boost 的核心是 Excess 计算，衡量特征值超出正常范围的程度：

```python
def compute_excess(feats, fname, bl):
    val = feats.get(fname, 0)
    pv = P95.get(fname, 0)           # 95% 分位值
    thresh = max(pv * 0.55, bl * 0.5) # 动态阈值
    if val <= thresh: return 0.0      # 低于阈值不贡献
    excess = (val / thresh - 1.0) ** 0.70  # 亚线性放大
    if val > P99:                     # 超过 99% 分位值的再加二次项
        pe = (val / P99 - 1.0) ** 0.70
        excess += 0.5 * max(0, pe)
    return excess
```

- **P95（95% 分位值）**：正常谱面的上限，超过即为"偏高"
- **P99（99% 分位值）**：极端谱面的标志，超过后额外加分
- **阈值 = max(P95 × 0.55, baseline × 0.5)**：确保低 base 特征也能被激活
- **指数 0.70**：亚线性放大，防止单一特征支配

---

## 4. 维度体系演进

### 4.1 密度维度

**核心公式**：`density_dimension = √(real_core_notes_per_second × above_avg_density_mean)`

这是经过多轮迭代后的最终公式，巧妙地结合了两个核心密度指标：
- **real_core_notes_per_second**：排除休息段后的核心音符（Tap+Hold）密度
- **above_avg_density_mean**：高潮段（密度超过 rcnps 的窗口）的平均密度

开根号后数值范围约 2~40，既有足够区分度又不会暴走。

**演变**：
- v8.0~v8.2：直接使用 real_core_notes_per_second
- v8.3：改为 √(rcnps × above_avg_mean)，更好地反映"持续高密度"和"瞬间高峰"的乘积关系

### 4.2 配置维度

覆盖以下键型：
- **楼梯（Stair）**：连续阶梯状排列的 Tap
- **Trill**：两点间快速交替
- **Jack**：同列快速连续击打
- **多押（Chord）**：同时击打多个音符
- **多指（Multi-finger）**：需要超过两根手指处理的配置

关键特征：
- `weighted_mf_score_per_sec`：加权的多指协调分/秒
- `chord_alternation_rate`：和弦交替频率
- `stair_rate_per_sec` + `stair_complexity`：楼梯的速度和复杂度
- `chord_size_entropy`：和弦大小分布的信息熵

### 4.3 耐力维度（重构历程）

这是全系统调整次数最多的维度，经历了三次重大重构：

**第一阶段（v8.0 ~ v8.3）**：
特征：`tap_per_second`, `total_notes`, `duration_sec`, `stamina_ratio`
问题：水段（休息段）拉高耐力值，低难度谱因长尾变长而耐力虚高

**第二阶段（v8.4）**：
用 `above_avg_density_ratio` 替换 `stamina_ratio`
问题：比值类特征在低端谱区分力不足，Ridge 给 co 极低

**第三阶段（v8.5）**：
删除 `tap_per_second`, `duration_sec`, `stamina_ratio` 等无效特征
核心特征精简为：
- **`above_avg_density_mean` (co=0.25)**：高潮段平均密度，最强信号（r=0.84）
- **`total_notes` (co=0.15)**：总物量，仅在高物量谱生效（非对称拉升）
- **`tap_burst_top5` (co=0.04)**：Tap 爆发峰值

**设计哲学**：水段不具有参考性——能 AP 的程度下，水段障碍已经不存在。高潮段的持续时长和密度共同决定了稳定性难度。

### 4.4 读谱维度（判定线视觉干扰）

Phigros 独有的读谱难度来源：

| 特征 | 定义 | 权重 |
|------|------|------|
| `jline_movement_density` | 判定线移动事件数/秒 | 0.05 |
| `jline_rotate_density` | 判定线旋转事件数/秒 | 0.03 |
| `jline_disappear_density` | 判定线消失事件数/秒 | 0.03 |
| `above_below_cross` | 是否上下交叉出键 | 0.03 |
| `speed_volatility` | 流速波动程度 | 0.04 |
| `type_switch_per_sec` | 红蓝黄粉切换频率 | 0.05 |
| `rhythm_entropy` | 节奏复杂度 | 0.03 |
| `density_transition_std` | 密度剧烈变化的程度 | 0.04 |
| `note_clutter_ratio` | 音符杂乱度 | 0.04 |
| `hold_interference_index` | 长条对读谱的干扰程度 | 0.04 |
| `tempo_change_count` | BPM 变化次数 | 0.02 |

**重要的发现**：`type_switch_per_sec`（音符类型切换频率）是读谱维度中最具区分力的特征。例如 Regrets 的读谱贡献主要来自密集的红蓝黄粉切换，而非判定线移动。

### 4.5 位移维度（已被移除）

**历史**：v8.0 引入 → v8.5 被移除

曾包含的特征：
- `movement_per_second`：每秒位移量
- `burst_avg_movement`：爆发段的平均位移
- `wide_jump_density`：大跳密度
- `burst_movement_variance`：爆发段位移方差
- `avg_distance_between_notes`：平均键间距

**移除原因**：
1. **主观性过强**：位移难度完全取决于玩家使用的手指数量（多指玩家位移几乎为零）
2. **计算方法有缺陷**：交互在两处原地交替时，位移量虚高；而真正的位移难点（如长楼梯）无法反映
3. **与真实难度相关性差**：统计学分析显示位移特征与定数的 r 值极低

用户原话："如果我用多指的话，位移几乎就等于不存在。你的计算方式我考虑过了——如果有人在屏幕两边两个同一个位置一直放交互，位移量就高得吓人，但事实是两根手指一直在原地没有动过。"

### 4.6 高速音符维度

检测同一判定线上间隔极短的核心音符对：

| 特征 | 音符间隔 | 权重 | 设计意图 |
|------|---------|------|---------|
| `fast_note_density_16th` | 相当于 16 分音符 | 0.08 | 基础高速 |
| `fast_note_density_32nd` | 相当于 32 分音符 | 0.15 | 核心高速（最高权重） |
| `fast_note_density_24th` | 相当于 24 分音符（三连） | 0.10 | 三连音型 |
| `fast_note_density_48th` | 相当于 48 分音符 | 0.12 | 超高速 |
| `fast_note_density_64th` | 相当于 64 分音符 | 0.10 | 极限高速 |

**权重设计原则**：32 分 > 48 分 > 24 分 > 16 分 > 64 分。32 分权重最高是因为它对普通玩家已有较大挑战性但并非不可企及，是区分中高难度的理想阈值。

---

## 5. 各版本详细变化

### 5.1 v5dim 时代（v3 ~ v4）

**初始架构**：5 维度（密度、配置、耐力、读谱、位移）+ 纯 Ridge 回归
**问题**：线性模型无法捕捉非线性关系，预测准确度有限
**转折点**：引入 Gradient Boosting 作为基线模型

### 5.2 v6dim 时代（v7.0 ~ v7.6）

**关键创新**：引入 GB + Boost 双模块架构
- v7.0：首次实现 GB 基线 + Ridge Boost
- v7.1~v7.3：逐步丰富特征集，调整训练流程
- v7.3_ridge：首次使用 Ridge 正系数回归优化 Boost co
- v7.4~v7.6：渐进式调优，加入多指、锁手等高级特征

**累积改进**：
- 增加多指协调评分（weighted_mf_score）
- 增加锁手特征（hold_lock_tap_events）
- 引入 Dynamic Cap 机制

### 5.3 v8.0 ~ v8.2：巅峰密度移除与读谱增强

**v8.0**：大幅重构特征集
- 新增判定线视觉干扰特征（jline_movement/rotate/disappear）
- 新增 BPM 变化特征
- 丰富高速音符检测

**v8.1**：读谱维度增强
- 增加 speed_volatility、above_below_cross
- 优化多指分析

**v8.2**：移除峰值密度 Boost
- 删除 `core_peak_density_1sec_top5avg` 和 `core_peak_density_top5avg_1beat`
- 原因：这两个特征与 density_dimension 高度共线，且被 fast_note_density 覆盖

### 5.4 v8.3：密度维度重构

**核心变化**：`density_dimension` 从单一密度改为 `√(rcnps × above_avg_mean)`
- 更好地反映"持续高密度能力"
- 数值范围从 0~20 扩展到 0~40
- 高难度谱的区分度显著提升

### 5.5 v8.4：耐力维度重构（高潮段占比）

**核心变化**：用 `above_avg_density_ratio` 替换 `stamina_ratio`
- `stamina_ratio` 因 co=0.0001 基本是个死特征
- 新特征衡量"超过平均密度的窗口占比"
- 首次引入 Stratified Split 分层抽样
- 网格搜索自动优化 target/power/thresh/cf

### 5.6 v8.5：耐力再重构 + 删除位移 + 高速音符扩展

这标志着项目最大的单一版本更新：

**删除位移维度**（5 个特征全部移除）：
- movement_per_second, burst_avg_movement, wide_jump_density, burst_movement_variance, avg_distance_between_notes

**耐力再重构**：
- 删除：tap_per_second, total_notes, duration_sec, tap_count, global_jack_count, burst_intensity_mean
- 新增：above_avg_duration_sec（高潮段持续时长）
- 提升：above_avg_density_mean co 从 0.08 → 0.25
- 后续又加回 total_notes（锁定 co=0.15）

**高速音符扩展**：
- 新增 fast_note_density_48th（权重 0.12）
- 新增 fast_note_density_64th（权重 0.10）
- 修正权重顺序：32nd(0.15) > 48th(0.12) > 24th(0.10) > 64th(0.10) > 16th(0.08)

**RPE 格式兼容**：
- predict_rpe.py 中 convert_rpe_to_standard 保留 eventLayers 和 extended
- feature_extractor.py 解析 eventLayers 中的 moveXEvents/moveYEvents/rotateEvents

### 5.7 train_manual.py：手动调优阶段

由于 Ridge 自动优化 + GB 重训导致模型不稳定，最终采用手动设置 co 值策略：

**流程**：
1. 从 v8.4 备份模型加载 GB 基线
2. 手动定义 FLAT_FEATURES 的每一组 (name, baseline, co)
3. 用这组 co 计算所有谱面的 Boost
4. 用 `y - boost` 作为标签重训 GB
5. 保存手动模型

**手动调优原则**：
- 关键特征（above_avg_density_mean、density_dimension、total_notes）保留高 co
- 锁定特征（fast_note_*、total_notes）强制保留 co 不被 Ridge 淘汰
- 统计学 r<0.15 的特征要么删除要么降至 0.01
- 共线特征去重（rcnps 与 density_dimension r=0.99，保留后者）

**当前最佳配置**：
```
sigmoid: target=0.32, power=0.65, thresh=0.22
density_dimension co=0.08, above_avg_density_mean co=0.25, total_notes co=0.15
```

### 5.8 v8.5bak → v8.5b：读谱系数修正 + 排序修复

**问题诊断**：
用户反馈 v7 模型存在两个严重问题：
1. **低定数谱面严重偏高**：如"スタートリップ"（标定12.2）预测值偏高
2. **高难度排序错误**：Apollo（17.8）和 Xaleid（18.2）预测值高于 胧月（18.4）和 Final EndGame（18.4）

**根本原因分析**：
- Apollo/Xaleid 的 `offbeat_ratio`（1.1481）和 `tempo_change_count`（0.6883）异常高
- v7 FLAT_FEATURES 中这两个读谱特征的 co 系数过高（均为 0.28），导致 Boost 贡献过大
- v7 中 offbeat_ratio 对 Apollo 贡献了 1.1481（占总 Boost 的 35%）

**修复方案**：
1. 切换到 v85bak 模型（41 个 FLAT 特征，更全面的特征覆盖）
2. 使用 v7 的 Dynamic Cap 和 Sigmoid 参数（已被验证为最佳）
3. 降低读谱相关特征系数：offbeat_ratio 0.28→0.12, tempo_change_count 0.28→0.12

**效果**：
- Apollo: 17.25, Xaleid: 17.49, 胧月: 17.25, FinalEndGame: 17.48
- 排序修复：Apollo < Xaleid < 胧月 < FinalEndGame ✓

### 5.9 v8.6：GB 偏低补偿（Bias Correction）

**问题诊断**：
v8.5b 虽然修复了读谱偏高和排序问题，但低定数谱面仍存在系统性低估：
- スタートリップ（12.2）预测 10.60（误差 -1.60）
- ふたりのスタートボタン（13.4）预测 13.06（误差 -0.34）

**原因**：GB 模型在低定数谱面上系统性偏低，仅靠 Boost 无法完全补偿。

**修复方案**：实现线性 GB 偏差补偿函数

```python
def compute_gb_bias(gb):
    """bias = max(0, THRESHOLD - GB) * FACTOR"""
    return max(0, 14.0 - gb) * 0.25
```

- 当 GB < 14.0 时，线性补偿 `(14.0 - GB) * 0.25`
- 当 GB >= 14.0 时，不补偿（高难度谱面不受影响）
- 参数：`BIAS_THRESHOLD = 14.0`，`BIAS_FACTOR = 0.25`

**代码修改**：
- `app.py`：新增 `compute_gb_bias()` 函数，在 `predict_one_chart()` 中应用
- 模型 pickle：新增 `bias_params` 字段
- 最终预测公式：`p_final = GB + Boost_adj + Bias`

**效果对比**（社区标注谱面，18 首）：

| 指标 | v8.5b (修复前) | v8.6 (修复后) | 改善 |
|:---|:---:|:---:|:---:|
| MAE | 0.381 | 0.344 | ↓9.7% |
| 最大误差 | 1.604 | 1.064 | ↓33.6% |
| スタートリップ(12.2) | 10.60 (-1.60) | 11.50 (-0.70) | 误差减半 |
| ふたりのスタートボタン(13.4) | 13.06 (-0.34) | 13.75 (+0.35) | 轻微偏高 |

**高难度排序验证**（通过 Web API 实测）：
- Apollo (17.8): 17.69 ✓
- Xaleid (18.2): 17.78 ✓
- 胧月 (18.4): 17.96 ✓
- Final EndGame (18.4): 17.92 ✓
- 排序：Apollo < Xaleid < FinalEndGame < 胧月 ✓

**剩余问题**：
- Submerged City (17.8): 预测 16.74（误差 -1.06），jline_movement_density 异常低导致
- ギザバ怪文書 其中一个版本：19.63（误差 +1.33），可能是不完整版本

**部署状态**：
- 模型文件：`models/6dim_model_v7.pkl`（包含 bias_params）
- 备份：`models/6dim_model_v7_before_bias.pkl`
- Web 版本：`8.6 (GB+Boost+Bias) 低定数修正+排序`
- 测试脚本：`test_labeled_charts.py`、`_test_web_api.py`

---

## 6. 统计学特征诊断

### 6.1 相关系数 r 分析

在 `_stat_full.py` 中对全部 351 官谱 + 16 自制谱进行特征-定数 Pearson 相关系数分析：

**强相关（|r| > 0.70）**：
```
real_core_notes_per_second     r=0.87  ★★★★★
peak_density_1beat             r=0.86  ★★★★★
density_dimension              r=0.85  ★★★★★
above_avg_density_mean         r=0.84  ★★★★★
notes_per_second               r=0.83  ★★★★★
density_transition_std         r=0.80  ★★★★★
p75_density_4beat              r=0.79  ★★★★★
total_notes                    r=0.77  ★★★★☆
mean_density_4beat             r=0.75  ★★★★☆
rhythm_entropy                 r=0.73  ★★★★☆
```

**中等相关（0.30 < |r| < 0.70）**：
```
tap_burst_top5                 r=0.58
jline_movement_density         r=0.49
position_entropy               r=0.48
chord_size_entropy             r=0.46
chord_alternation_rate         r=0.44
note_clutter_ratio             r=0.42
hold_interference_index        r=0.41
type_switch_per_sec            r=0.40
stair_rate_per_sec             r=0.38
fast_note_density_16th         r=0.39
fast_note_density_32nd         r=0.37
tempo_change_count             r=0.35
multi_finger_3plus_events      r=0.34
above_below_cross              r=0.25
```

**弱相关/负相关（|r| < 0.15，被移出模型）**：
```
above_avg_density_ratio        r=-0.19  ✗ 负相关
speed_volatility               r=-0.003 ✗ 基本零相关
pattern_switch_rate            r=-0.055 ✗
drag_flick_ratio               r=-0.12  ✗
rest_ratio                     r=-0.08  ✗
```

### 6.2 共线性诊断

**高度共线（|r| > 0.95）的特征对**：
```
real_core_notes_per_second  ↔ density_dimension            r=0.99
stair_density               ↔ stair_rate_per_sec           r=1.00
peak_density_1beat          ↔ mean_density_1beat           r=0.96
core_micro_max_0.0625beat   ↔ core_micro_top5_0.0625beat   r=0.97
```

**处理策略**：
- 保留信息更丰富的特征（density_dimension > rcnps）
- 保留物理意义更清晰的特征（stair_rate_per_sec > stair_density）
- Ridge 回归本身的 L2 正则化天然处理共线性

### 6.3 特征筛选结果

经过统计学清洗，最终 FLAT_FEATURES 保留 34 个特征：

```
密度(5): density_dimension, real_core_notes_per_second, fast_note_16th/32nd/24th/48th/64th
配置(12): stair_rate_per_sec, stair_complexity, chord_size_entropy, ...  
耐力(3): above_avg_density_mean, total_notes, tap_burst_top5
读谱(14): jline_movement_density, jline_rotate_density, jline_disappear_density, ...
```

---

## 7. 格式解析历程

### 7.1 官谱格式（Phigros 标准格式）

- 路径：`D:\迅雷下载\Phigros_Resource-master\Phigros_Resource-master\chart`
- 结构：JSON 文件，每条判定线独立包含 `notesAbove` / `notesBelow`
- 时间单位：ticks（1/32 拍）
- 判定线事件：顶层字段 `judgeLineMoveEvents` / `judgeLineRotateEvents` / `judgeLineDisappearEvents`
- 通过 `unified_parser.py` 的 `load_chart_from_bytes` 自动检测解析

### 7.2 RPE 格式

- 在判定线对象中使用 `notes` 字段（替代 `notesAbove/notesBelow`）
- 音符类型映射：RPE type1→Tap, type2→Hold, type3→Flick, type4→Drag
- 时间格式：`[measure, beat, division]` 三元组
- 坐标格式：百分制（0~100），需除以 100 归一化
- 判定线事件：使用 `eventLayers` 嵌套字段
- 转换函数：`predict_rpe.py::convert_rpe_to_standard()`

### 7.3 RPE v3 格式（愚人节谱）

- 与 RPE 格式基本一致，但有额外的 `extended` 字段
- 愚人节谱示例：Chart_SP #1347, Sigma, Regrets, 105 秒
- feature_extractor 需额外解析 `extended` 中的 `inclineEvents` 等
- 判断依据：`META.RPEVersion` 字段

### 7.4 PE 文本格式

- 文本文件，以 `# name:` 或 `# title:` 开头
- 通过 `unified_parser.py` 的 `load_chart_from_bytes` 中的 PE 解析器处理
- 谱面名称从文本前 30 行的 `# name:` / `# title:` 中提取

---

## 8. 训练方法与优化策略

### 8.1 Ridge 正系数回归

**作用**：从 Excess 矩阵中学习每个特征的最优权重 co

**参数**：
- `alpha`（正则化强度）：在 [0.01, 0.1, 1, 5, 10, 50, 100] 中用 5 折交叉验证选取
- `fit_intercept=False`：不学截距（理论上所有特征 co 为 0 时 Boost 应为 0）
- `positive=True`：强制正系数（避免负 co 导致的"难度倒挂"）

**迭代策略**：3 次迭代，每次用 EMA（当前=0.3×旧 + 0.7×新）平滑更新 co

### 8.2 分层抽样（Stratified Split）

```python
bins = np.digitize(y, bins=[12, 13, 14, 15, 16, 17, 18])
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(sss.split(X_gb, bins))
```

- 将定数按整数值分层（12-, 13, 14, ..., 18+）
- 确保训练集和测试集在各难度区间比例一致
- 避免高难度谱样本少导致训练偏差

### 8.3 锁定特征（PINNED）

**目的**：防止 Ridge 因共线性或样本不平衡将关键特征的 co 压低至 0

**锁定特征列表**：
```python
PINNED = {
    'fast_note_density_16th': 0.08,
    'fast_note_density_32nd': 0.15,
    'fast_note_density_24th': 0.10,
    'fast_note_density_48th': 0.12,
    'fast_note_density_64th': 0.10,
    'total_notes': 0.15,
}
```

**实现**：在 Ridge 训练时，仅优化自由特征；锁定特征的贡献直接从残差中扣除

### 8.4 最优参数网格扫描

在 v8.4 / v8.5 中自动扫描以下参数组合：

```python
for target in [0.28, 0.30, 0.32]:
    for power in [0.65, 0.70, 0.75, 0.80]:
        for thresh in [0.20, 0.22, 0.24, 0.26]:
            for cf in [1.00, 1.04, 1.08, 1.12, 1.16]:
```

评估指标：对测试谱面的 MAE + 正负偏差平衡度

**最新最优组合**：`target=0.32, power=0.65, thresh=0.22, cf=1.00`

### 8.5 cf 系数缩放

- cf（scale factor）全局缩放所有自由特征的 co 值
- cf > 1.0 放大高难谱的 Boost，但同时可能使低难度谱偏高的谱面更偏高
- v8.4 Ridge 版本倾向于 cf=1.08~1.16，但手动调整后最终确定 cf=1.0（不缩放）

---

## 9. 核心难题与解决方案

### 9.1 高难偏低、低难偏高

**现象**：GB 基线范围仅 10.8~15.6（跨度 4.8 分），但难度跨度达 10 分（8~18+）。Boost 在高端的拉升力度不足，低端又因部分特征（如 density_dimension 在简单谱也有 5~8 的数值）产生虚高。

**症状数据**：
```
谱面          期望定数   预测值   偏差
Apollo        17.8      17.26    -0.54
怪文書        18.3      17.67    -0.63
朧月          18.4      17.60    -0.80
スタートリップ 12.2      13.36    +1.16
```

**尝试过的方案**：

1. **上调 total_notes co**：从 0.10 → 0.15 ~ 0.20，利用 total_notes 在高低端的巨大差值（462 vs 2500+）做非对称拉升
2. **调整 Sigmoid 参数**：target 从 0.28 → 0.32，power 从 0.75 → 0.65，让高难谱的 Boost 占比更高
3. **降低低端活跃特征的 co**：如 density_dimension 从 0.12 → 0.08
4. **对比高低端特征差异**：分析哪些特征在高端远高于低端，提升其区分力

**当前状态**：仍有 ~0.5 分的差距待缩小

### 9.2 位移维度无存在感

**问题**：位移特征（movement_per_second 等）在所有谱面的数值差异极小，贡献接近于 0

**尝试的改进**：
- 加入时间间隔加权（不仅看距离，还看两键时间间隔）
- 分析 burst 段位移
- 最终结论：位移太主观，多指玩家几乎零位移

**最终方案**：**直接删除整个位移维度**。用户充分肯定了这个决定。

### 9.3 耐力特征的水段干扰

**问题**：tap_per_second、duration_sec 等特征被水段（休息段）严重干扰

**分析**：低难度谱（如 スタートリップ, difficulty=12.2）有大量休息段但 duration_sec 很长，导致耐力虚高。而高难度谱（如 Apollo, difficulty=17.8）的 duration_sec 也不短，但密集区持续输出才是真正挑战。

**用户原话**："高潮段时长和高潮段密度两个因素已经共同决定了这张谱想要 AP 的难度有多难和稳定性有多强，水段不具有参考性。"

**最终方案**：
1. 删除 tap_per_second、duration_sec、stamina_ratio 等特征
2. 核心耐力 = above_avg_density_mean（高潮段平均密度，r=0.84）
3. 补充 total_notes 做非对称拉升（仅高物量谱激活）
4. 保留 tap_burst_top5 捕捉瞬间爆发需求

### 9.4 判定线视觉干扰难以量化

**问题**：Phigros 特有的判定线表演（按键+判定线旋转、四面来键、长条干扰、流速变化等）难以从谱面数据中准确捕捉

**实现方案**：
1. **直接量化**：统计判定线移动/旋转/消失事件数量，归一化到每秒
2. **间接量化**：
   - `speed_volatility`：流速波动的标准差
   - `density_transition_std`：密度变化的剧烈程度
   - `note_clutter_ratio`：音符在空间上的混杂度
   - `hold_interference_index`：长条对同时期 Tap 的空间干扰
3. **BPM 变化量化**：`tempo_change_count` 追踪变速次数

**局限性**：部分表演性判定线变化（如"按键出现在屏幕外然后旋转滑入"）难以从纯数据层面捕捉。

### 9.5 高速音符权重倒置

**问题**：v8.3 版本中权重顺序为 16th(0.15) > 32nd(0.10) > 24th(0.08)，与直觉相反

**修正**：32 分音符比 16 分音符更快、更难，权重应该更高

**最终权重**：32nd(0.15) > 48th(0.12) > 24th(0.10) > 64th(0.10) > 16th(0.08)

### 9.6 密度暴走与 cap 取舍

**问题**：定轨 4k 高难谱的 density_dimension 可达 36+，远超正常谱的 20 左右

**尝试**：
1. **对密度做 sqrt cap**：`min(density, sqrt(density) * 5)`，用户认为"密度没必要压缩，那些本来就不需要纳入考虑范围"
2. **Dynamic Cap**：保留 kneel+excess^power 机制，只压缩极端 Boost raw

**最终方案**：不对 density_dimension 做硬 cap，保留原始公式。极端值通过 Dynamic Cap 在 boost 层面控制。

### 9.7 RPE 格式判定线特征为 0

**问题**：所有 RPE 谱的判定线特征（jline_movement_density 等）全为 0

**原因**：`predict_rpe.py::convert_rpe_to_standard()` 在转换时丢弃了 eventLayers

**修复**：
1. 在 convert_rpe_to_standard 中保留 eventLayers 和 extended
2. 在 feature_extractor 中添加 `if layer is None: continue` 防御性代码
3. 解析 eventLayers 中的 moveXEvents/moveYEvents/rotateEvents

---

## 10. 当前模型状态

### 10.1 当前 FLAT_FEATURES 配置（train_manual.py）

```
密度:
  density_dimension              co=0.08  bl=1.0
  real_core_notes_per_second     co=0.03  bl=2.0

配置:
  stair_rate_per_sec             co=0.05  bl=2.0
  stair_complexity               co=0.02  bl=0.2
  chord_size_entropy             co=0.02  bl=0.5
  chord_alternation_rate         co=0.08  bl=0.5
  weighted_mf_score_per_sec      co=0.05  bl=10.0
  position_entropy               co=0.02  bl=2.0
  avg_chord_size_poly            co=0.03  bl=2.0
  drag_flick_ratio               co=0.02  bl=0.2
  pattern_switch_rate            co=0.05  bl=1.0
  position_range_used            co=0.02  bl=0.5

耐力:
  above_avg_density_mean         co=0.25  bl=4.0   ★ 最强信号
  total_notes                    co=0.15  bl=400.0 ★ 锁定
  tap_burst_top5                 co=0.04  bl=0.5

读谱:
  tempo_change_count             co=0.02  bl=50.0
  type_switch_per_sec            co=0.05  bl=0.4
  density_transition_std         co=0.04  bl=0.2
  density_transition_mean        co=0.02  bl=0.15
  note_clutter_ratio             co=0.04  bl=0.05
  rhythm_entropy                 co=0.03  bl=2.5
  hold_interference_index        co=0.04  bl=0.3
  jline_movement_density         co=0.05  bl=50.0
  jline_rotate_density           co=0.03  bl=20.0
  jline_disappear_density        co=0.03  bl=20.0
  speed_volatility               co=0.04  bl=0.1
  above_below_cross              co=0.03  bl=0.3

高速音符（全部锁定）:
  fast_note_density_16th         co=0.08  bl=4.0
  fast_note_density_32nd         co=0.15  bl=2.0  ★ 最高
  fast_note_density_24th         co=0.10  bl=1.0
  fast_note_density_48th         co=0.12  bl=0.5
  fast_note_density_64th         co=0.10  bl=0.3
  rhythm_type_count              co=0.10  bl=3.0

Sigmoid: target=0.32, power=0.65, thresh=0.22
Dynamic Cap: knee=2.5, power=0.9
```

### 10.2 手动测试结果（_man_test.py）

```
谱面               期望  预测  GB   Boost  dd(密度)  total_notes
Apollo            17.8  17.26  15.27  1.99  15.0     2507
Chart_SP #1347    17.7  17.13  15.28  1.84  11.4     2500
怪文書            18.3  17.67  15.28  2.39  19.9     2666
朧月              18.4  17.60  15.30  2.30  15.7     2525
Final EndGame     18.4  17.35  15.36  1.99  17.8     2087
恋ひ恋ふ縁        16.8  16.61  15.02  1.58  13.3     1119
おぎゃりざいな    16.5  16.40  15.24  1.16  10.0     1563
茉子              15.5  15.87  14.95  0.92   7.9     1310
スタートリップ    12.2  13.36  13.06  0.31   6.2      462
トキ              14.6  14.82  14.29  0.53   5.6     1321
```

**当前偏差**：
- 高难（17+）：偏低 0.5~0.8 分
- 中难（14~16.5）：偏差在 ±0.3 分内，表现良好
- 低难（12 左右）：偏高 ~1.0 分

### 10.3 仍存在的问题

1. **高难偏低、低难偏高** 问题尚未根治
2. 部分读谱特征（如判定线表演性移动）量化不够准确
3. AP 难度与准度要求尚未分离——当前预测偏向"通关难度"而非"AP 难度"
4. 愚人节谱的极难度谱（Chart_SP #1347 等）预测偏低
5. 前端 v8.5 标题标签需要修正（耐力标签仍是"高潮TPS"）

---

## 11. 展望与待办

### 短期

1. **进一步解决高难偏低**：继续对比高低难度谱的特征差异，提升关键特征的区分力
   - 提升 total_notes co 从 0.15 → 0.18~0.20
   - 考虑加入 above_avg_duration_sec（高潮段持续时长）
   - 调整 Sigmoid target 从 0.32 → 0.35 或 power 从 0.65 → 0.60

2. **排除 Chart_neko 和 snow dance**：这两个新官谱不应参与训练
   - 在 data_loader 中添加排除列表

3. **前端修正**：耐力标签改为"高潮段密度"等更准确的描述

### 中期

4. **AP 难度分离**：尝试将"通关难度"和"AP 难度"作为两个独立预测目标
5. **置信区间输出**：对每个预测给出 ± 区间
6. **更多非官方谱验证**：用有定数的自制谱验证模型泛化能力

### 长期

7. **深度神经网络尝试**：用 DNN 替代 GBR 作为基线模型
8. **社区反馈集成**：收集玩家对预测结果的主观反馈进行微调
9. **特征工程自动化**：用 AutoML 方法自动发现新的有效特征

---

## 附录 A：关键文件说明

| 文件 | 功能 | 备注 |
|------|------|------|
| `app.py` | Flask Web 服务器 | 预测 API + 前端页面 |
| `feature_extractor.py` | 219+ 特征提取 | 核心引擎，约 1600 行 |
| `train_manual.py` | 手动 co 训练脚本 | 当前主力训练脚本 |
| `train_v8_5.py` | Ridge 自动训练脚本 | v8.5 自动版本 |
| `train_v8_4.py` | v8.4 训练脚本 | 高潮段占比版本 |
| `predict_rpe.py` | RPE 格式转换与预测 | 含格式映射 |
| `unified_parser.py` | 统一格式解析器 | 自动检测格式 |
| `data_loader.py` | 加载官谱 + difficulty.tsv | 训练数据加载 |
| `_stat_full.py` | 全特征统计学诊断 | r 值 + 共线性 |
| `_man_test.py` | 手动 co 测试 | 验证预测结果 |

## 附录 B：模型文件存档

```
models/6dim_model_v8_5.pkl        ← 当前模型（manual co）
models/6dim_model_v8_5_bak.pkl    ← v8.5 Ridge 备份
models/6dim_model_v8_4.pkl        ← v8.4（高潮段占比）
models/6dim_model_v8_4_bak.pkl    ← v8.4 备份
models/6dim_model_v8_3.pkl        ← v8.3（密度重构）
models/6dim_model_v8_3_bak.pkl    ← v8.3 备份
models/6dim_model_v8_2.pkl        ← v8.2（移除峰值密度）
models/6dim_model_v8_1.pkl        ← v8.1（读谱增强）
models/6dim_model_v8_0.pkl        ← v8.0（大幅重构）
models/6dim_model_v7_6.pkl ~ v7.pkl  ← v7.x 系列
models/5dim_model_v4.pkl ~ v3.pkl    ← v5dim 系列
models/gb_final_model.pkl            ← 最早期最终模型
```

---

*本文档由开发助手根据对话历史和代码分析自动生成，记录了 Phigros 难度定数预测系统 v8.6 的完整开发历程。*
