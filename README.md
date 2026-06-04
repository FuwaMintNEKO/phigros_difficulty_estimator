# Phigros 难度定数预测系统

基于梯度提升回归(GradientBoostingRegressor)的Phigros谱面定数预测系统，支持官谱/RPE/RPE v3三种格式自动检测与解析。

## 模型版本

| 版本 | MAE | 训练数据 | 核心改进 |
|------|-----|----------|----------|
| v7.1 | 0.589 | 957官谱 | 维度均衡(配置×0.55/读谱×2.0)，sigmoid压缩 |
| v7.2 | 0.589 | 957官谱 | BPM解析修复(per-line BPM、RPE变速保留) |
| **v7.3** | **0.406** | **349 IN/AT官谱** | **Ridge数据驱动co学习 + 迭代重训GB** |

## 快速开始

```bash
pip install flask numpy scikit-learn
python app.py
# 访问 http://127.0.0.1:5000
```

## 项目结构

```
├── app.py                 # Flask Web应用入口
├── feature_extractor.py   # 特征提取（5维：密度/位移/配置/耐力/读谱）
├── unified_parser.py      # 统一谱面解析器（官谱/RPE/RPE v3自动检测）
├── predict_rpe.py         # RPE→标准格式转换
├── data_loader.py         # 官谱数据加载
├── train_6dim_v7_3.py     # v7.3训练脚本（Ridge迭代优化）
├── templates/
│   └── index.html         # 前端界面
├── models/
│   ├── 6dim_model_v7_3.pkl          # v7.3模型（当前使用）
│   └── 6dim_model_v7_3_backup.pkl   # 备份
└── 📂 _analyze_*.py       # 分析/诊断脚本
```

## 预测原理

### 特征体系（5大维度）

| 维度 | 代表特征 | 含义 |
|------|----------|------|
| 密度 | `density_dimension = √(持续TPS × 峰值TPS)` | √(真实密度 × 爆发密度) |
| 平均位移 | `movement_per_second` | 判定线移动幅度 |
| 配置 | `stair_density`, `chord_size_entropy`, `multi_finger_3plus` | 键型复杂度 |
| 耐力 | `stamina_ratio`, `tap_per_second`, `total_notes` | 体力消耗 |
| 读谱 | `density_transition_mean`, `offbeat_ratio`, `rhythm_entropy` | 读谱难度 |

### 预测流程

```
谱面JSON → 特征提取 → GB基线预测 → Boost加成 → sigmoid压缩 → 最终定数
```

- **GB** (GradientBoosting): 从219个特征预测基线难度
- **Boost**: 对超出P95阈值的特征累加贡献（co由Ridge从IN/AT官谱学习）
- **Sigmoid**: 当boost/GB比值过高时平滑压缩（target=0.24, power=0.70, thresh=0.24）

## 谱面格式支持

| 格式 | 说明 | 自动检测规则 |
|------|------|------------|
| 官谱/Standard | 标准Phigros JSON | `judgeLineList`包含`notesAbove/notesBelow` |
| RPE普通 | RPE编辑器格式 | `META.RPEVersion`存在 |
| RPE v3 | 愚人节单线谱 | 某线notes>800且有移动事件 |
| PE格式 | 纯文本谱面 | 不以`{`开头 |

## 更新日志

### v7.3 (当前)
- Ridge数据驱动：从IN/AT官谱学习最优co值，替代手调权重
- 迭代重训GB：co更新后重新拟合GB残差，3轮收敛
- MAE: 0.406（测试集17谱，正偏8/负偏8）

### v7.2
- 修复`predict_rpe.py`：RPE转换时保留BPMList
- 修复`feature_extractor.py`：无BPMList时每条线使用自己的BPM
- 修复`_parse_bpm_timeline`：兼容float格式startTime
- 前端BPM显示改为范围（如Apollo: 180~339）

### v7.1
- 维度均衡因子：配置×0.55, 读谱×2.0等
- Sigmoid平滑压缩引入

## 网页部署

```bash
# 开发模式
python app.py

# 生产模式（Linux）
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

上传谱面文件（支持.json/.zip批量），点击"开始预测"即可。
