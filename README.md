# Phigros 难度定数预测系统

基于梯度提升回归(GradientBoostingRegressor) + Ridge学习boost权重的Phigros谱面定数预测系统。

## 当前版本: v8.1

**MAE: 0.341** (17个测试谱, IN/AT官谱训练)

## 快速开始

```bash
pip install flask numpy scikit-learn
python app.py
# 访问 http://127.0.0.1:5000
```

## 预测原理

```
谱面JSON → 5维特征提取 → GB基线预测 → Boost加成 → sigmoid压缩 → 最终定数
```

### 5大维度

| 维度 | 代表特征 | 含义 |
|------|----------|------|
| 密度 | `density_dimension = √(持续TPS × 峰值TPS)` | 综合真实密度和爆发力 |
| 平均位移 | `movement_per_second` | 判定线移动幅度 |
| 配置 | `fast_note_density_16th`, `avg_chord_size_poly`, `rhythm_type_count` | 键型与节奏复杂度（**v8新增**） |
| 耐力 | `stamina_ratio`, `tap_per_second`, `total_notes` | 体力消耗 |
| 读谱 | `density_transition_mean`, `offbeat_ratio`, `rhythm_entropy` | 读谱难度 |

- **GB** (GradientBoosting): 219特征预测基线难度(max_depth=5, 700树)
- **Boost**: 44个excess特征累加贡献，co由Ridge从349张IN/AT官谱迭代学习
- **Sigmoid**: 当boost/GB比值过高时平滑压缩(tgt=0.26, pwr=0.60, thr=0.22)

### v8新增配置特征

| 特征 | P95 | corr | 含义 |
|------|-----|------|------|
| `fast_note_density_16th` | 5.38 | +0.84 | 同线≤16分音次数/秒（BPM归一化） |
| `avg_chord_size_poly` | 2.34 | +0.59 | 多押事件平均note数 |
| `rhythm_type_count` | 8 | +0.61 | 唯一节奏类型数 |

这3个特征替代了旧版5个弱特征（jack_density, sim_pos_spread_mean, position_cluster_count, track_deviation_score, drag_flick_ratio）。

### v8.1 BPM变速积分修复

`time_to_seconds(tick, bpm)` 旧实现假设全曲恒定BPM，在变速谱上全错。v8.1重写为BPM timeline逐段积分，所有绝对时间调用点修复。

## 谱面格式支持

| 格式 | 说明 | 自动检测 |
|------|------|----------|
| 官谱/Standard | 标准Phigros JSON | `judgeLineList`包含`notesAbove/notesBelow` |
| RPE普通 | RPE编辑器格式 | `META.RPEVersion`存在 |
| RPE v3 | 愚人节单线谱 | 某线notes>800且有移动事件 |
| PE格式 | 纯文本谱面 | 不以`{`开头 |

## 倍速功能

前端滑块 0.5~2.0x，拖动后 debounce 400ms 自动重预测：
- BPM与判定线速度等比缩放
- GB锁定1x基线，仅boost随速度变化
- rest_gap阈值自动随速度缩放（`1.0/speed`）

## 项目结构

```
├── app.py                 # Flask Web应用入口
├── feature_extractor.py   # 5维特征提取 + BPM变速积分
├── unified_parser.py      # 统一谱面解析器（官谱/RPE/RPE v3）
├── predict_rpe.py         # RPE→标准格式转换
├── data_loader.py         # 官谱数据加载
├── train_v8.py            # v8.x训练脚本（Ridge迭代优化）
├── templates/
│   └── index.html         # 前端界面
├── models/
│   ├── 6dim_model_v8_1.pkl          # v8.1模型（当前）
│   ├── 6dim_model_v8_0.pkl          # v8.0模型
│   ├── 6dim_model_v7_3.pkl          # v7.3模型
│   └── 6dim_model_v7_3_backup.pkl   # v7.3备份
└── _*.py                  # 分析/诊断/调试脚本
```

## 测试谱预测结果 (v8.1)

| 谱面 | 定数 | 预测 | 偏差 |
|------|------|------|------|
| 怪文書 | 18.3 | 18.32 | +0.02 |
| Apollo | 17.8 | 18.02 | +0.22 |
| 胧月 | 18.4 | 18.00 | -0.40 |
| Xaleid◆scopiX | 18.2 | 17.99 | -0.21 |
| The Final EndGame | 18.4 | 17.92 | -0.48 |
| Waking Shadows | 17.8 | 18.50 | +0.70 |
| silly-willy-nilly | 17.7 | 17.67 | -0.03 |
| Submerged City | 17.8 | 17.48 | -0.32 |
| Cheerio! | 17.0 | 17.27 | +0.27 |
| 恋ひ恋ふ縁 | 16.8 | 17.22 | +0.42 |
| Lemegeton | 16.6 | 16.61 | +0.01 |
| トキラキメキ | 14.6 | 14.62 | +0.02 |
| ふたりのスタートボタン | 13.4 | 13.49 | +0.09 |
| スタートリップ | 12.2 | 10.77 | -1.43 |

## 完整版本历史

### v8.1 — BPM变速积分修复 (2025-06-05)
- 修复 `time_to_seconds`: 从恒定BPM → BPM timeline逐段积分
- 影响所有变速谱的时长/密度/TPS特征
- 修复 feature_extractor 中10个绝对时间调用点
- MAE: 0.341
- 键盘谱: FinalEndGame +0.21, 胧月 +0.22

### v8.0 — 新配置特征 + 精简boost (2025-06-05)
- 新增3个boost配置特征: fast_note_density_16th, avg_chord_size_poly, rhythm_type_count
- 删除5个弱/冗余boost特征: jack_density等
- GB不变(219特征), 新特征仅加入boost
- MAE: 0.334 (v7.3=0.353)
- sigmoid: target=0.24, power=0.60, thresh=0.22

### v7.3 — Ridge数据驱动 + BPM修复 (2025-06-04)
- 休息段显示修复: 改为`时长-有效时长`(含头尾死区)
- Ridge从349张IN/AT官谱迭代学习boost权重(3轮)
- BPM解析修复: per-line BPM、RPE变速保留、BPMList兼容
- 删除 `_dynamic_cap` 的MM曲线, 改用指数衰减
- MAE: 0.353

### v7.2 — BPM解析修复 (2025-06-04)
- RPE转换保留BPMList
- 无BPMList时每条判定线使用自己的BPM
- `_parse_bpm_timeline` 兼容float格式startTime
- 前端BPM显示改为范围

### v7.1 — 5维均衡 (2025-06-04)
- 5大维度均衡: 配置×0.55/读谱×2.0/密度×1.2/位移×1.1
- 统一密度: `√(总TPS×峰值)`
- 型识别、多指协调、位置熵
- sigmoid boost压缩 (target=0.24, power=0.70, thresh=0.24)
- 批量上传支持

### v5.3 — 指数衰减Cap (2025-06-03)
- 指数衰减cap替代MM曲线, 高难谱不再被过度压缩
- GB: 600树, max_depth=5, lr=0.05

### v5.2 — 初始版本 (2025-06-03)
- GradientBoostingRegressor + 5维度非线性boost
- 统一模型(不分IN/AT)
- 225个特征, P95/P99阈值基于957张官谱
- Boost cap 3.5
