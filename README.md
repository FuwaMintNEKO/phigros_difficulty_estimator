# Phigros 难度定数预测系统

基于 **Gradient Boosting 残差 + 手工 Boost 叠加 + Level one-hot** 的 Phigros 谱面难度定数（Chart Constant）预测器。

以 **官方谱面定数为唯一权威基准**（982 张官谱训练），内推精准预测自制谱定数，外推（≥17.7）保持社区定级趋势。

## 当前版本: v11.2（+ v11.3 speed 统一）

- **官谱歌曲分组 5 折 CV MAE: 0.5200**（AT 段 MAE 0.385，历史最优）
- **官谱 in-sample MAE: 0.0086**（权威标尺无损）
- **上架自制谱 589 张**：14-15 段 +0.14 / 15-16 +0.13 / 16-17 +0.01 / ≥17 -0.28

## 快速开始

```bash
pip install flask numpy scikit-learn pandas
python app.py
# 访问 http://127.0.0.1:5000
```

## 预测原理

```
谱面JSON → 特征提取(256维) → GB残差基线(含level one-hot) → Boost叠加(条件缩放) → 定轨段加成 → 校准 → 最终定数
```

### 核心机制

| 模块 | 说明 |
|---|---|
| **GB 残差** | 256 特征 + 3 类 level（EZ/HD/IN_AT，IN/AT 合并已验证更优）+ 低段加权 1.5 + 尾杀特征 |
| **Boost 叠加** | 48 个手工权重特征（密度/配置/耐力/读谱/位移/差速/和弦重键），excess 指数 0.70，cap 4.0 |
| **密度去冗余（v11.2）** | above_avg_density_mean 按"有效击打数"（同押去冗余，4k 全押只计 1 次），修复多押撑密度虚高 |
| **条件缩放（v11）** | 仅自制谱：多指谱压 mf 特征（多面型重压/多押型重压/楼梯型轻压），双指谱抬 eff 有效单指密度（wmf 平滑过渡 12~18） |
| **定轨段加成（v11.1）** | 4k/5k/6k 键盘段检测（2.5s 窗口聚类槽位数），占比归一化加成 |
| **预测时校准** | 仅自制谱：14-15 -0.30 / 15-16 -0.18 / 16-17 -0.05 |
| **密度域对齐** | 自制谱 IN 段密度特征向官谱分布对齐（data/domain_align.json） |
| **倍速（v11.3）** | speed 参数 = BPM 缩放后全量特征预测（与"改 BPM 的 json"完全一致）；wmf 堆料档平滑化消除倍速跳变 |

### 谱面格式支持

| 格式 | 说明 |
|---|---|
| 官谱/Standard | 标准 Phigros JSON（formatVersion:3） |
| RPE | PhiEdit 导出（META.RPEVersion，isFake 假音符已过滤，eventLayers 变速已合并） |
| RPE v3 | 愚人节单线谱（双条件判定：numOfNotes + META.RPEVersion） |
| PE 文本 | PhiEditer 遗留（时间单位=拍×32，type 映射已修正） |

## 版本历史

| 版本 | 核心变化 | 官谱 CV |
|---|---|---|
| v5dim | 5 维特征 + Ridge | — |
| v6dim v7.x | GB 基线 + Boost 双模块架构 | — |
| v8.x | 读谱/配置/耐力重构，删除位移维度 | — |
| v10.1 | Level one-hot（3 类 IN/AT 合并）+ 尾杀 + 低段加权 | 0.5264 |
| v11.0 | chord_size_entropy 负熵 bug 修复 | 0.5227 |
| v11.1 | 定轨段特征（4k/5k/6k） | 0.5247 |
| **v11.2** | **密度去冗余（方案B）+ 双指堆料档 + 多面重压 + 阈值重标** | **0.5200** |
| **v11.3** | **speed 统一为改 json 行为 + wmf 档位平滑 + domain align 双倍 bug 修复** | — |

## 模型文件

| 文件 | 说明 |
|---|---|
| models/6dim_model_v11_2.pkl | 当前生产模型（v11.2） |
| models/6dim_model_v11_2_stable.pkl | v11.2 稳定备份 |
| models/6dim_model_v11_1_stable.pkl | v11.1 备份 |
| models/6dim_model_v11_stable.pkl | v11.0 备份 |
| models/6dim_model_v10_before_v11.pkl | v10.1 备份 |

## 项目结构

```
├── app.py                 # Flask Web 入口（预测 API + 前端）
├── feature_extractor.py   # 256 维特征提取（含定轨段/有效单指密度）
├── unified_parser.py      # 统一谱面解析器（官谱/RPE/RPE v3/PE）
├── boost_config.py        # Boost 手工权重配置（48 特征）
├── data_loader.py         # 官谱数据加载
├── predict_rpe.py         # RPE → 标准格式转换
├── data/
│   ├── chart/             # 官谱（982 张，不入库）
│   ├── phira/             # 自制谱数据（上架 589 / 未上架 957+ / 4.4星 6006）
│   └── info/difficulty.tsv  # 官谱定数表（权威标签）
├── docs/                  # 开发历史 / 审计报告 / 版本报告
├── train/                 # 训练脚本（train_v11_a.py 等）
└── tools/                 # 实验/分析脚本
```

## 关键文档

- docs/估算偏高问题审计与修复.md — 全部实验与审计记录（30+ 节）
- docs/v11模型改良报告.md — v11/v11.2/v11.3 改良报告
- docs/PHIGROS_DIFFICULTY_ESTIMATOR_DEV_HISTORY.md — v8.6 前开发历程
- project_memory.md — 跨会话项目记忆（关键结论/未决事项）

## 注意事项

- 训练集 = 官谱 982 张（3.11.0+ 标度，最高 Rrhar 17.6）；预测 >17.8 为标度外推
- 社区定数 ≠ 官谱标尺：16+ 段社区普遍偏高（多指谱尤甚），双指谱偏低——模型以官谱为准
- 变速欺诈谱（2xBPM 等）预测偏差是已知系统性盲区，只提示不修正
- 自制谱特殊 level（ST/EX/SP 等）统一按 IN/AT 处理（3 类模型）
