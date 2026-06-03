# Phigros 难度定数预测器

基于 **Gradient Boosting + 特征外推 (Boost)** 混合模型的 Phigros 谱面难度预测系统。

**核心思路：** GB 模型学习"常规"难度映射，Boost 机制捕捉极端谱面的尾部难度，两者叠加得到最终定数。

## 快速开始

### 环境要求

- Python 3.10+
- pip

### 安装

```bash
# 克隆仓库
git clone https://github.com/FuwaMintNEKO/phigros_difficulty_estimator.git
cd phigros_difficulty_estimator

# 安装依赖
pip install -r requirements.txt
```

依赖只有三个：`Flask`、`numpy`、`scikit-learn`。

### 启动 Web 服务

```bash
python app.py
```

默认监听 `http://0.0.0.0:5000`。可用参数：

```bash
python app.py --host 127.0.0.1 --port 8080     # 改地址和端口
python app.py --debug                          # 调试模式
```

### 使用

1. 浏览器打开 `http://127.0.0.1:5000`
2. 拖拽谱面 JSON 文件（官谱/RPE 格式）或 ZIP 压缩包到上传区
3. 支持批量上传，点击「开始预测」
4. 展开详情可查看 **5 大类别贡献值**、**原始特征值**、**特征明细表格**

![webui](screenshot_webui.png)

### 特征说明

页面直方图显示两类数值：

| 位置 | 含义 | 示例 |
|------|------|------|
| **上方数值** | 该类别的 Boost 贡献值 | `0.44` |
| **下方数值（含单位）** | 该类别主特征的原始值 | `8.56 键/秒 (TPS)` |

5 大类别：

| 类别 | 核心特征 | 单位 |
|------|---------|------|
| 密度 | 核心音符密度 (tap+hold) | 键/秒 |
| 1smax密度 | 1 秒窗口核心音符峰值 | 键/秒 |
| 平均位移 | 每秒位移量 | 格/秒 |
| 耐力 | tap 密度 + stamina 比 | 键/秒 |
| 读谱 | 节奏复杂度 + BPM 变化 | - |

## 模型架构

```
输入谱面 JSON → 特征提取 (240维) → GB 预测残差 + 特征外推 Boost
                                                   ↓
                                       动态 Cap (MM曲线)
                                                   ↓
                                              最终定数
```

- **GB 模型：** GradientBoostingRegressor (600 棵树, max_depth=5, lr=0.05)
- **动态 Cap：** 指数衰减，`knee=2.5, power=0.9`（超出部分开 0.9 次方，无硬上限）
- **训练数据：** 957 张官谱
- **特征数量：** 240 个原始特征 → 28 个平铺特征
- **测试表现：** 20 张测试谱平均误差 0.193，15/20 < 0.3

### API

除了 Web UI，还提供 JSON API：

**`POST /predict`** — 批量预测（multipart/form-data）
```bash
curl -F "files=@chart.json" http://127.0.0.1:5000/predict
```

**`POST /predict_one`** — 单谱预测（application/json，供 Android 悬浮窗使用）
```bash
curl -X POST http://127.0.0.1:5000/predict_one \
  -H "Content-Type: application/json" \
  -d @chart.json
```

## 手机部署（Termux）

参考 [deploy_termux.sh](deploy_termux.sh) 一键部署，或在 Termux 中：

```bash
pkg update -y
pkg install -y python python-pip
pip install Flask numpy scikit-learn
git clone https://github.com/FuwaMintNEKO/phigros_difficulty_estimator.git
cd phigros_difficulty_estimator
python app.py --host 0.0.0.0 --port 5000
```

## 重新训练

如果你有自己的谱面数据集想重新训练：

```bash
# 确保 data_loader.py 配置了正确的数据路径
python train_5dim_v4.py
```

训练完成后会自动保存新模型到 `models/5dim_model_v5_3.pkl`。

## 测试评估

```bash
python evaluate_standalone.py
```

会输出 22 个测试谱面的预测结果与预期值对比表。

## 项目结构

```
├── app.py                   # Flask Web 服务器（主入口）
├── feature_extractor.py     # 特征提取模块（240维）
├── predict_rpe.py           # RPE 谱面格式解析
├── unified_parser.py        # 统一谱面解析器
├── data_loader.py           # 数据加载
├── train_5dim_v4.py         # 训练脚本（当前版本 v5.3）
├── evaluate_standalone.py   # 测试评估脚本
├── requirements.txt         # Python 依赖
├── deploy_termux.sh         # Termux 手机部署脚本
├── templates/
│   └── index.html           # Web 前端
└── models/
    └── 5dim_model_v5_3.pkl  # 当前模型
```

## 更新日志

### v5.3 (2026-06-03)
- **动态 Cap 重构**：将 Michaelis-Menten 曲线替换为**指数衰减策略**。
  - 低于 knee (2.5) 的部分保持线性
  - 超出部分 `excess ^ 0.9` 加到结果上，无硬上限
  - 真正的高难谱面（如 People people、Galaxy Collapse）不再被过度压缩
- **训练 R²=0.9564** (v5.2: 0.9535) | **训练 MAE=0.6735** (v5.2: 0.695)
- **测试 20 谱平均误差=0.193**，15/20 < 0.3

### v5.2 (2026-06-03)
- 密度特征从 NPS 改为 TPS（核心音符：tap+hold）
- 旧密度退化为辅助小特征
- 训练 R²=0.9535
```
