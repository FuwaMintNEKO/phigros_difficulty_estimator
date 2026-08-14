## v8.13 部署完成 (2026-06-06)

### 最终架构: XGBoost 纯模型

经过 v8.6~v8.12 多轮迭代，最终确定 XGBoost 为最优方案。

### 版本对比总览

| 版本 | 模型 | 架构 | 测试MAE | 结论 |
|:---|:---|:---|:---:|:---|
| v8.5 | Ridge | 手动系数 + Boost修正 | 0.341 | 手工系数难调 |
| v8.6 | GB | GB基线 + Boost修正 | 0.0001 | **过拟合** |
| v8.7 | GB | GB主导 + Boost条件修正 | 0.3427 | Boost贡献50-80% |
| v8.7b | GB | 纯GB (无Boost) | 0.3427 | 与v8.7持平 |
| v8.8 | Ridge | Ridge + Boost校正 | 0.3502 | 线性模型表达力不足 |
| v8.9 | GB | GB + Boost作为特征 + 网格搜索 | 0.3335 | Boost重要性220/220 |
| v8.10 | GB | 纯GB + 最优超参 | 0.3408 | 验证Boost无用 |
| v8.11 | XGBoost | 纯XGBoost | 0.3283 | **优于GB** |
| v8.12 | Ensemble | GB+XGB加权平均 | 0.3270 | 边际改善，复杂度高 |
| **v8.13** | **XGBoost** | **纯XGBoost + 最优参数** | **0.3283** | **最终部署** |

### 关键发现

1. **Boost对树模型无增益**：v8.9中Boost特征重要性排名220/220，v8.10纯GB MAE=0.3408与v8.9的0.3335几乎持平，差异来自超参数调整
2. **XGBoost优于GB**：相同输入下，XGBoost测试MAE=0.3283 vs GB=0.3392，提升约3.2%
3. **Ensemble边际改善有限**：GB+XGB加权平均MAE=0.3270，仅比纯XGB提升0.0013，不值得增加复杂度
4. **特征重要性Top 10**：real_notes_per_second(0.17), peak_tps_1sec(0.16), notes_per_second(0.08), core_peak_density(0.05), above_avg_density_mean(0.04)... 密度/速度类特征主导

### 最优超参数

```python
XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42
)
```

### 剩余问题

极端谱面系统性低估（测试集）：
- Credits.Frums: 15.7 → 14.53 (-1.17)
- opia.rN: 15.6 → 14.41 (-1.19)
- CervelleConnexion: 14.5 → 13.66 (-0.84)

这些谱面在Phigros社区中定数可能存在争议，或模型缺少视觉复杂度、判定线特效干扰等特征。

### 部署

- 模型文件: `models/6dim_model_v8_13.pkl`
- app.py 已更新：XGBoost直接预测，Boost仅用于前端展示
- 服务器: `http://127.0.0.1:5000`