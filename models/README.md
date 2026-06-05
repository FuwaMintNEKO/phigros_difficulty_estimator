# 模型文件说明

## 当前模型

| 文件 | 版本 | MAE | 说明 |
|------|------|-----|------|
| `6dim_model_v8_1.pkl` | **v8.1 (当前)** | 0.341 | BPM变速积分修复 + 新配置特征 |
| `6dim_model_v8_0.pkl` | v8.0 | 0.334 | 新配置特征（fast_16th/avg_chord/rhythm） |
| `6dim_model_v7_3.pkl` | v7.3 | 0.353 | Ridge数据驱动权重 |

## 备份

| 文件 | 说明 |
|------|------|
| `6dim_model_v7_3_backup.pkl` | v7.3最早备份 |
| `6dim_model_v7_3_backup2.pkl` | v8训练前备份 |
| `6dim_model_v7_3_original.pkl` | v7.3原始副本 |

## pickle内容

| 字段 | 类型 | 说明 |
|------|------|------|
| `gb` | `GradientBoostingRegressor` | GB模型（700树, max_depth=5） |
| `scaler` | `StandardScaler` | 特征标准化器 |
| `feature_names` | `list[str]` | 219个GB特征名 |
| `p95_vals` | `dict[str,float]` | 各特征P95阈值 |
| `p99_vals` | `dict[str,float]` | 各特征P99阈值 |
| `FLAT_FEATURES` | `list[(str,float,float)]` | boost特征 (name, baseline, co) |
| `dynamic_cap` | `dict` | 指数衰减cap参数 (knee=1.0, power=0.90) |
| `sigmoid_params` | `dict` | sigmoid压缩参数 (target, power, thresh) |
