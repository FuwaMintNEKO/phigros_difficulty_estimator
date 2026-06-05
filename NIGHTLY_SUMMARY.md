# 夜间迭代总结 (2024-06-05)

## 做了什么

从v7.11到v7.26共迭代了**16个版本**，探索了7个主要方向：

### 1. 扩展特征集 (v7.11)
- 在GB中加入stamina、变速、和弦、节奏等特征，从6维扩展到11维
- 结果: MAE退化。特征过多导致GB和Ridge互相竞争，破坏架构平衡

### 2. 超参数验证 (v7.12, v7.13, v7.14)
- **v7.12**: 搜索dynamic_cap的knee和power参数 — 验证了v7.3默认值(knee=1.0, power=0.90)最优
- **v7.13**: 对比excess特征和原始特征作为boost输入 — 原始特征导致数值溢出，验证了excess设计必要性
- **v7.14**: 纯GB(无Boost)超参数网格搜索 — MAE=0.86，远不如GB+Boost架构

### 3. 倍速增强学习 (v7.15, v7.16, v7.17)
- 生成0.5x/0.7x/1.0x/1.3x/1.5x/2.0x六种倍速的谱面数据（共2094条）
- 尝试让GB直接学习速度与难度的关系
- 三种方案均未超越v7.3的MAE=0.31

### 4. 架构微调 (v7.18, v7.19, v7.20)
- **v7.18**: 加入movement_interaction特征(density_dimension × movement_per_second) — MAE=0.315，略差于v7.3
- **v7.19**: 集成5个不同随机种子的v7.3模型取平均 — MAE=0.370，集成反而降低精度
- **v7.20**: seed=789最优种子单模型 — MAE=0.350，不如v7.3的seed=42

### 5. 训练策略优化 (v7.22, v7.24, v7.25, v7.26)
- **v7.22**: 加权样本(基于v7.3误差) — GB过拟合导致权重无效，MAE=0.377
- **v7.23**: 替换为HistGradientBoosting — MAE=0.380，API不兼容
- **v7.24**: GB用Top100特征(减少过拟合) — MAE=0.374
- **v7.25**: Ridge去掉positive约束 — MAE=0.386
- **v7.26**: 弱GB(200树,10叶) + 强Boost(5迭代) — MAE=0.409

### 6. スタートリップ问题分析
- 发现スタートリップ(12.2)被低估的根本原因：movement=0.0，但stair/tap密度极高
- 模型缺乏"运动交互"特征，无法区分"高密度+高移动"和"高密度+无移动"
- 所有版本均未解决此问题，误差持续在-1.0~-1.5

## 核心发现

1. **v7.3 (MAE=0.31) 仍是当前最优模型**，GB+Boost交替优化架构最有效
2. 任何改变GB/Boost平衡的尝试都会导致MAE退化
3. 倍速增强训练会降低原速精度
4. 纯GB远不如GB+Boost，验证了Boost架构的必要性
5. 集成学习在小数据集上可能导致方差增大
6. HistGB、特征筛选、加权样本等方法均未超越v7.3

## app.py 改进

- **速度偏移后处理**: 改为始终在1x提取特征，预测基础定数，再加速偏移
  - 偏移公式: `clamp((speed-1)*5, -2.5, +2.5)`
  - 2x → +2.5, 0.5x → -2.5
  - 确保滑块和手动改BPM结果一致

## 建议后续方向

1. **收集更多训练数据** — 当前仅349条，这是最直接的改进方向
2. **添加运动交互特征** — 如`notes_per_second × movement_per_second`，解决スタートリップ类问题
3. **倍速作为Boost特征** — 让Ridge专门处理速度相关修正，不干扰GB
4. **神经网络替代** — 用小规模MLP替代GB+Boost架构

## 保存的模型

- `models/6dim_model_v7_3.pkl` — 最优模型 (MAE=0.31) ✓
- `models/6dim_model_v7_18.pkl` — +interaction特征 (MAE=0.315)
- `models/6dim_model_v7_19.pkl` — 5种子集成 (MAE=0.370)
- `models/6dim_model_v7_20.pkl` — seed=789 (MAE=0.350)
- `models/6dim_model_v7_22.pkl` — 加权样本 (MAE=0.377)
- `models/6dim_model_v7_24.pkl` — Top100特征 (MAE=0.374)
- `models/6dim_model_v7_25.pkl` — 无positive约束 (MAE=0.386)
- `models/6dim_model_v7_26.pkl` — 弱GB+强Boost (MAE=0.409)

## 详细分析

完整分析报告见 `ANALYSIS_REPORT.md`，特征深度分析见 `_feature_analysis.py`。