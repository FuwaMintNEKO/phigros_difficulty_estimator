## v8.20 最终总结 (2026-06-06)

### 最终模型: XGBoost v8.13

经过完整的模型迭代（v8.5~v8.20），最终确定 **XGBoost** 为最优方案。

---

### 完整模型对比

| 版本 | 模型 | 测试MAE | 结论 |
|:---|:---|:---:|:---|
| v8.5 | Ridge + Boost修正 | 0.3410 | 手工系数难调 |
| v8.6 | GB + Boost | 0.0001 | **过拟合** |
| v8.7 | GB主导 + Boost条件修正 | 0.3427 | Boost贡献50-80% |
| v8.7b | 纯GB | 0.3427 | 与v8.7持平 |
| v8.8 | Ridge + Boost校正 | 0.3502 | 线性模型表达力不足 |
| v8.9 | GB + Boost特征 + 网格搜索 | 0.3335 | Boost重要性220/220 |
| v8.10 | 纯GB + 最优超参 | 0.3408 | 验证Boost对树模型无用 |
| v8.11 | 纯XGBoost | 0.3283 | **优于GB** |
| v8.12 | GB+XGB Ensemble | 0.3270 | 边际改善(+0.0013) |
| **v8.13** | **XGBoost + 最优参数** | **0.3283** | **最终部署** |
| v8.14 | LightGBM | 0.3518 | 劣于XGBoost |
| v8.15 | XGBoost变体 (Huber等) | 0.3283+ | Baseline已最优 |
| v8.16 | 特征选择/加权训练 | 0.3265+ | 边际改善 |
| v8.17 | Stacking残差模型 | 0.3238 | 边际改善，增加复杂度 |
| v8.18 | 极端谱面诊断 | — | in-sample拟合极好，问题在泛化 |
| v8.19 | MLP神经网络 | 2.6966 | 完全失败（样本不足） |
| v8.20 | 分段模型 | 0.2718 | 有潜力但需预测时选段 |

### 最优超参数

```python
XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42
)
```

### 特征重要性 Top 10
1. real_notes_per_second (0.169)
2. peak_tps_1sec (0.158)
3. notes_per_second (0.080)
4. core_peak_density (0.050)
5. above_avg_density_mean (0.041)
6. peak_density_1sec (0.024)
7. bpm_mean (0.022)
8. note_count (0.018)
9. duration_seconds (0.016)
10. click_ratio (0.015)

### 关键发现

1. **Boost对树模型完全无用**: v8.9中Boost特征重要性排名220/220；v8.10纯GB MAE=0.3408与v8.9的0.3335差异来自超参数
2. **XGBoost > GB > LightGBM**: 相同输入下，XGBoost比GB提升3.2%，比LGB提升6.7%
3. **Ensemble边际改善有限**: 仅0.0013，不值得增加复杂度
4. **神经网络不适用**: 350样本远不足以训练MLP（MAE=2.7）
5. **分段模型有潜力**: MAE=0.2718但需解决预测时选段问题

### 剩余问题

极端谱面系统性低估（测试集）：
- Credits.Frums (15.7): 预测14.53 (-1.17)
- opia.rN (15.6): 预测14.41 (-1.19)
- CervelleConnexion (14.5): 预测13.66 (-0.84)
- Nhelv (15.6): 预测14.83 (-0.77)

这些谱面可能具有现有特征无法捕获的难度因素（视觉复杂度、判定线特效、pattern复杂度等），或Phigros社区定数存在争议。

### 部署信息

- 模型文件: `models/6dim_model_v8_13.pkl`
- 服务器: `http://127.0.0.1:5000`
- 模型类型: XGBoost (MODEL_TYPE='xgboost')
- 特征数: 299
- 训练数据: 350谱面 (全量训练)