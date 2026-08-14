# project_memory.md — 跨会话项目记忆

> 维护: 每次会话结束更新关键状态与未决事项

## 当前状态（2026-08）

- **生产模型**: `models/6dim_model_v11_2.pkl`（v11.2: 密度去冗余方案B + 双指堆料档 + 多面重压 + 条件boost + 校准）
- **稳定备份**: `models/6dim_model_v11_2_stable.pkl`、`6dim_model_v11_1_stable.pkl`、`6dim_model_v11_stable.pkl`、`6dim_model_v10_before_v11.pkl`
- **模型版本线**: v5dim → v6dim(v7.x) → v8.x → v10.1 → v11.0 → v11.1
- **权威基准**: 官谱定数（difficulty.tsv, 982张, 3.11.0+标度, 最高17.6）; 上架谱社区定数为第二权威测试集; 未上架谱仅测试
- **官谱 CV**: 歌曲分组5折 MAE **0.5200** / in-sample MAE 0.0086
- **上架谱(589)**: 14-15 +0.14、15-16 +0.13、16-17 +0.01、≥17 -0.28

## v11.1 新增（2026-08, 本次会话）

### 定轨键盘段特征（4k/5k/6k, 用户提出）
- **概念**: 4k/5k/6k ≠ 同时押数, 而是定轨音游玩法（固定槽位密集击打, 多指分工, 双指无解）
- **实现**: feature_extractor.py `compute_track_segments_features()`: 2.5s滑动窗口聚类positionX(间距>=1.5), 主导槽(槽内>=4音符)数=k; 输出 tracks_4plus_sec/5plus/6plus/max_k/avg_k/active_sec
- **效果**: Chart_SP(16.79→**17.11**)含大量4k(20-60s)/5k(70-120s)/6k(120-130s)段; 官谱 corr: active_sec +0.857、max_k +0.788
- **推理层加成**(app.py, 仅自制谱, 校准前): +0.15*min(r4,0.8) + 0.55*min(r5,0.4) + 1.0*min(r6,0.15), r=各轨段时长占比; 上架谱平均加成仅+0.038(97%谱<0.15)
- 定轨特征进GB池(256特征)后Chart_SP仍不足17 → 靠推理层加成解决(GB对OOD极端值外推有限, avg_k=4.02超官谱max 3.4)

### 其他
- test_charts 扩充到 41 张（从 Downloads 转移 25 张标定谱）
- 删除 2xBPM 倍速谱

## v11 关键结论（勿重复实验）

1. **训练层改 boost 权重无效**（GB 完全补偿）— 只有推理层修改有净效果
2. **AT/16+ 段样本加权无法修 bias**（GB 外推保守, 加权被补偿）
3. **chord_size_entropy 已修复**（负熵bug+漏单押）— Spearman -0.028→+0.539
4. 条件 boost: 多指 mf×0.50(低密度)/×0.70(高密度≥12.5), 双指 eff×1.5, 仅自制谱
5. 预测时校准: (14,15,-0.30),(15,16,-0.18),(16,17,-0.05), 仅自制谱
6. **多押只算tap+hold**（用户确认, flick/drag同押不算多押）

## 已否决方向（不重复投入）

- 混合训练、训练时标签校准、isotonic校准、Huber/log1p、协同/位移/底力交互特征、flick计入多押、2xBPM谱修正（只提示不修正）

## 特征不平衡研究（2026-08）

| 不平衡点 | 状态 |
|---|---|
| 定轨键盘段(4k/5k/6k)难度缺失 | ✅ v11.1已加（tracks_*特征+推理层加成） |
| chord_size_entropy 负熵/漏单押 | ✅ v11已修复 |
| 多指谱OOD外推虚高/双指耐力低估 | ✅ v11条件boost已校正 |
| 变速欺诈谱(2xBPM等) | 已知盲区, 只提示不修正（社区同样处理不好） |
| 可馅蜜/容错、前段白送 | 无特征, 官谱无标签, 记录不追 |
| 尾杀窗口细化(5%/3s) | 方案就绪(tools/notes_tail_window.md), 优先级低 |
| 判定线表演性读谱 | 部分量化(jline_*), 演出型仍难 |

kyou标签投票分析结论(11.7/11.9): 变速/闪现仍低估(-0.13), 面海+0.30高估; 核心拉分特征已被现有体系覆盖, 特征工程无需扩充(除定轨发现外)

## 未决事项

- [ ] 14-15/15-16 段残余 +0.17~0.19 高估（社区口径 vs 官谱标尺差异）
- [ ] 尾杀窗口细化候选（tools/notes_tail_window.md）
- [ ] 官谱 16+/11-13 段 OOF 低估 -0.2/-0.25（无干净解法）
- [ ] スタートリップ(12.2) 预测 10.95 低估（PE老谱, 特征域差异）
- [ ] 自定义level谱(ST/EX/FINAL)映射IN, 部分特殊谱偏差
- [ ] data/phira/feats_cache_v11.pkl 需在特征变更后重建（tools/build_feats_cache_v11.py）

## 实验脚本索引（v11.x）

| 脚本 | 用途 |
|---|---|
| tools/exp_v11_diag.py | 上架谱偏差诊断(生产口径) |
| tools/exp_v11_cv_bias.py | 官谱分组CV bias分析 |
| tools/exp_v11_conditional.py | 条件boost参数实验 |
| tools/exp_v11_final_sim.py | 最终方案模拟 |
| tools/exp_v11_production_verify.py | 生产路径综合验证 |
| tools/_sim_track_bonus.py | 定轨加成模拟 |
| tools/_dbg_chartsp*.py / _dbg_track*.py | 定轨分析调试 |
| tools/build_feats_cache_v11.py | 特征缓存构建 |
| train/train_v11_a.py | v11训练脚本(--atw/--boostvar/--out) |

## 数据文件

- data/phira/charts.json: 上架565+特殊50谱元数据
- data/phira/predictions.csv: 旧模型预测+社区定数(diff列)
- data/phira/neighbor_estimate.csv: 邻居法官谱标尺估计
- data/phira/json/: 615张上架谱面; json_unranked/: 957张未上架
- data/domain_align.json: 自制谱IN段密度域对齐delta
- data/test_charts/: 41张自制谱测试集（含Downloads转移25张）

## 未上架 4.4 星谱下载进度（2026-08）

- 元数据: data/phira/unranked_all.json（type=2 常规 8634 张全量）
- 4.4星+(rating>=0.88): 6006 张；含 regular 标签 100%（无污染）
- 目标目录: data/phira/json_unranked_4star/
- 续传脚本: tools/fetch_unranked_44star.py（8线程并发, 已有文件自动跳过, 可反复运行续传）
- 清单: data/phira/unranked_4star_list.csv
