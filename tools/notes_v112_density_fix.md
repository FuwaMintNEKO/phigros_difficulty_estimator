# v11.2 密度虚高修正方案（t2）

> 任务：审计 eff_peak_tps_1s / eff_avg_tps_1s 计算，设计"密度虚高修正"特征方案（方案A/B/C），用官谱982验证分布与定数相关性。
> 状态：方案设计完成，未修改 feature_extractor.py 主文件。
> 验证数据：官谱 982（feats_cache_v11.pkl official）+ 上架谱 589（ranked diff>10），模型 v11.1（6dim_model_v11_1.pkl）。

---

## 1. 背景与问题

部分自制谱靠"多押（4k 全押）/对拍"撑起高密度特征（above_avg_density_mean、real_core_notes_per_second 等），
导致密度虚高、预测偏高。已确认的问题谱例：

| 谱面 | nps | dens(above_avg) | eff_avg | eff_density_ratio | 特征 |
|---|---|---|---|---|---|
| ギザバ怪文書(18.3) | 16.98 | 22.37 | 12.06 | **0.539** | mf3=151, wmf=43.8 |
| Sigma Regrets | 25.43 | 21.13 | 5.92 | **0.280** | mf3=161, mf4=138 |

（Sigma Regrets 的 eff_avg/dens=0.28 与任务描述完全吻合；两谱 dens 远超官谱 >=17 段均值 13.11。）

---

## 2. 审计：eff_peak_tps_1s / eff_avg_tps_1s 计算（feature_extractor.py 773-801 行）

```python
# 1秒窗口内"独立击打次数": 窗口内音符按 tick 聚类, 同押组(tick差<1)只算1次
for right in range(len(cts_sorted)):
    while cts_sorted[right] - cts_sorted[left] > 1.0: left += 1
    seg = ctk_sorted[left:right + 1]
    if len(seg) >= 2:
        eff = 1 + int(np.sum(np.diff(seg) >= 1))   # 组数 = 1 + 相邻tick差>=1的次数
    else:
        eff = int(len(seg))
features['eff_peak_tps_1s'] = int(max_eff)           # 最大有效击打数
features['eff_avg_tps_1s']   = float(np.mean(eff_vals))  # 全部窗口均值
```

**审计结论：逻辑正确。**
- 同押判定：同一 tick（差=0）或相邻 tick（差<1，即 1/32 拍内）归为同一击打组，`1 + Σ(diff>=1)` 即组数。volcanic 4押海 1秒28音符 → 有效≈7；键盘连打 1秒27单点 → 有效=27，语义符合设计。
- 局限①：`eff_avg_tps_1s` 是**全部**滑动窗口（含低密度空隙）的有效击打均值，与 above_avg_density_mean（**高潮段**原始窗口 TPS）量纲与窗口定义均不同——直接相除得到 ratio 是"两种口径的混合"，但作为**冗余度指标**依然有效（验证见 §4）。
- 局限②：eff 只对 tap+hold 去冗余，drag/flick 不计入 core，符合设计。
- 现状：eff_ 两个特征**只在 boost 条件逻辑中使用**（EFF_FEATS：双指谱 eff_scale=1.5 抬升），**不在主特征集（MANUAL_FLAT/6dim 模型）中**，也未参与 above_avg_density_mean 的计算。

---

## 3. 密度虚高的根源：above_avg_density_mean（803-825 行）

```python
# 高潮段 = 1s窗口原始计数(同押算多次) >= rcnps 的窗口
if window_tps >= rcnps: above_windows.append(window_tps)
above_avg_density_mean = mean(above_windows)
```

窗口 TPS 用**原始音符计数**（4k 全押 1 窗口计 4 个音符），同押冗余直接撑高密度 → 虚高根源。
`eff_` 特征已提供去冗余信息，但未接入该公式。

---

## 4. 验证数据（官谱982）

### 4.1 特征与官谱定数 diff 的相关性

| 特征 | Pearson | Spearman |
|---|---|---|
| above_avg_density_mean | 0.9039 | 0.9527 |
| **eff_avg_tps_1s** | **0.9107** | 0.9454 |
| eff_peak_tps_1s | 0.8927 | 0.9466 |
| real_core_notes_per_second | 0.8836 | 0.9342 |
| **eff_density_ratio** | **-0.5346** | **-0.5644** |

结论：eff_avg_tps_1s 的 Pearson 甚至略高于 above_avg_density_mean——去冗余后密度信息并未损失。
eff_density_ratio 与定数**负相关**：官谱高难段本征多押多（volcanic/SATELLITE ratio≈0.52），不能直接作为正向特征。

### 4.2 eff_density_ratio 分布（官谱982）

| P5 | P25 | P50 | P75 | P90 | P95 |
|---|---|---|---|---|---|
| 0.629 | 0.717 | 0.776 | 0.847 | 0.912 | 0.959 |

按定数段（中位）：<13=0.813 → 13-14=0.756 → 14-15=0.733 → 15-16=0.718 → 16-17=0.696 → >=17=0.595（单调递减）。

### 4.3 ratio 与多押特征

- ratio vs weighted_mf_score_per_sec：**P=-0.7013 / S=-0.8160**（强负相关 → 低 ratio 确实=多押撑密度，设计意图成立）
- 原始 dens vs mf3：0.4100；eff_avg vs mf3：0.2832（去冗余后与多押相关性降 31%）

### 4.4 上架589 vs 官谱同段（关键：自制谱密度虚高的直接证据）

| 段 | 官谱 ratio 中位 | 上架 ratio 中位 | 上架 dens/官谱 dens | 上架 wmf/官谱 wmf |
|---|---|---|---|---|
| 13-14 | 0.756 | 0.718 | 0.95 | 1.19 |
| 14-15 | 0.733 | 0.719 | 1.00 | 1.19 |
| 15-16 | 0.718 | 0.712 | 1.03 | 1.16 |
| 16-17 | 0.696 | 0.682 | 1.05 | 1.08 |
| >=17 | 0.595 | 0.625 | 1.11 | 1.08 |

上架谱 ratio 系统性低于官谱同段、高段 dens 虚高 5-11%、多押强度全面更高 → **"密度虚高"在上架谱中真实存在**。

---

## 5. 方案A：eff_density_ratio 新特征

```python
eff_density_ratio = eff_avg_tps_1s / max(above_avg_density_mean, 0.1)
```

- 分布见 §4.2；多押撑密度谱 ratio 低（Sigma Regrets 0.28、ギザバ 0.54），真连打谱 ratio 高（接近 1.0）。
- **关键性质：与定数负相关（-0.53/-0.56）**，直接加入线性模型/FLAT 会系统性拉低高难段预测 → **不宜作为独立正向特征**。
- 推荐用法：作为**密度修正系数**（对 above_avg_density_mean 的贡献做平滑缩放，见方案C），或作为多押分支条件（_sim_v112d 已用 eff_ratio<0.40 做硬分支）。

## 6. 方案B：修改 above_avg_density_mean 公式（按有效单指）

```python
# 改动: 窗口 TPS 用"有效击打数"替代原始计数 (eff 聚类逻辑复用)
# 窗口内 seg 的有效击打 = 1 + Σ(diff(seg_tick) >= 1)
if eff_count >= eff_avg:      # 阈值同步改为 eff 版均值
    above_windows_eff.append(eff_count)
above_avg_density_mean = mean(above_windows_eff)
```

**精确重算验证（官谱982，复制逻辑与缓存 corr=1.0）**：
- 与 diff 相关性几乎不变：P 0.9039→**0.9045**，S 0.9527→0.9482
- 高难段降幅大（多押谱被正确压低）：>=17 降 18%，16-17 降 8.4%，15-16 降 5%
- 低难段略升 7.4%（阈值改用 eff_avg 后的窗口选择效应）
- 官谱典型：volcanic 14.9→8.94（-40%）、SATELLITE 11.3→7.15（-37%）、QZKagoRequiem 15.4→10.33（-33%）

**全量模拟（v11.1 模型未重训，仅替换 dens 值）**：
- 上架589 整体偏差 +0.040 → **-0.006**（各段 -0.04~-0.06）
- 副作用：官谱 16-17 段偏差 Δ-0.054、>=17 段 Δ-0.122（官谱高难段本征多押被误伤）→ **必须重新训练模型 + 重新校准**才能落地

**结论**：方案B 特征本身更干净、相关性保持，是长期最优解；代价是改变所有谱面值（重提缓存+重训），且单独替换（不重训）会误伤官谱高难段。

## 7. 方案C（推荐）：ratio 残差修正——相对官谱基准的密度修正

**思路**：官谱高难段 ratio 低是"本征多押"（真难），上架谱 ratio 低是"堆料"（虚难），特征层面无法直接区分 → 用**相对官谱同段基准的残差**衡量"虚高程度"，只修正虚高部分。

```python
# 官谱基准: 按定数段的 ratio 中位 (官谱982统计, §4.2)
RATIO_BASELINE = {'<13':0.811, '13-14':0.756, '14-15':0.733,
                  '15-16':0.718, '16-17':0.696, '>=17':0.595}
def ratio_baseline(diff):
    return RATIO_BASELINE['<13'] if diff<13 else \
           RATIO_BASELINE['13-14'] if diff<14 else \
           RATIO_BASELINE['14-15'] if diff<15 else \
           RATIO_BASELINE['15-16'] if diff<16 else \
           RATIO_BASELINE['16-17'] if diff<17 else RATIO_BASELINE['>=17']

def density_ratio_residual(diff_approx, feats):
    """方案C新特征: 有效单指密度比 - 官谱同段基准 (负=相对虚高)"""
    ratio = feats.get('eff_avg_tps_1s', 0) / max(feats.get('above_avg_density_mean', 0), 0.1)
    return ratio - ratio_baseline(diff_approx)   # diff_approx 预测时用 GB 基线 p_gb
```

**验证（上架589）**：
- 官谱残差对称（mean≈0，P25=-0.054，P75=+0.054）；**上架谱 55.9% 残差<0**（密度相对虚高），mean=-0.0127
- 残差与预测偏差方向完全一致：残差<-0.02（虚高）组平均偏差 **+0.083**（预测偏高）；残差>+0.02（扎实）组平均偏差 **-0.031**（预测偏低）
- 谱例：Daydream 残差-0.176 偏差+1.33；3rd Avenue -0.194 偏差+0.85；魔理沙 残差+0.221 偏差-2.53（真连打被低估）

**boost 集成方式（平滑，无硬分支，无 cap）**：

```python
# 在 _sim_v112d 的 DENS_FEATS 处理中:
dens = feats.get('above_avg_density_mean', 0)
effa = feats.get('eff_avg_tps_1s', 0)
ratio = effa / max(dens, 0.1)
resid = ratio - ratio_baseline(p_gb)          # p_gb = GB 基线预测
dens_corr = 1.0 + 1.2 * min(0.0, resid)        # 仅虚高(resid<0)打折, 扎实谱不动
# above_avg_density_mean 的 e 值乘以 dens_corr (e = v/t - 1)
```

**方案C优点**：
- 只修正"相对官谱虚高"部分，官谱回归不受影响（残差对称、均值 0）
- 无需重新提取特征缓存（eff_avg/above_avg 已在缓存中），即插即用
- 平滑连续、无硬分支、无 cap，符合 v11.2 "不加cap、不加协同" 原则
- 与多押条件（mf3/mf4 分支）正交：resid 是密度维度修正，多押分支管配置维度

**参数建议**：dens_corr 系数 1.0~1.5（先 1.2）；阈值用 0（残差>0 不动）避免正向放大；
平滑可选 `dens_corr = 1/(1+exp(1.2*(base-resid)))` 形式替换 min 硬折线。

---

## 8. 实现代码建议（汇总）

### 8.1 特征层（feature_extractor.py 新增，不动现有特征）

```python
# 在 eff_ 计算块之后 (798 行后) 新增:
features['eff_density_ratio'] = eff_avg / max(features.get('above_avg_density_mean', 0), 0.1)
```

### 8.2 boost 层（_sim_v112d.py 的 predict 中，DENS_FEATS 分支扩展）

```python
elif fname in DENS_FEATS:
    co2 = co * dens_scale
    if fname == 'above_avg_density_mean':
        ratio = effa / max(dens, 0.1)
        resid = ratio - ratio_baseline(p_gb)
        e = e * (1.0 + 1.2 * min(0.0, resid))   # 仅虚高打折
```

### 8.3 方案B（如需根治，feature_extractor.py 803-825 行）

```python
# above_avg 窗口计数改为有效击打数 (复用 eff 的 tick 聚类)
# 阈值从 rcnps 改为 eff_avg; 需重提缓存 + 重训模型
```

---

## 9. 预期影响

| 项 | 方案A(独立特征) | 方案B(改公式) | 方案C(残差修正) |
|---|---|---|---|
| 上架谱整体偏差 | 需调参 | +0.040→-0.006 | 预计 -0.03~-0.05 |
| 密度虚高谱（低ratio） | 需交互项 | 下降明显 | 下降 0.2~0.5 |
| 真连打谱（高ratio） | 需交互项 | 低难段+7% | 不动 |
| 官谱回归 | 高段被压 | 16-17/-0.05, >=17/-0.12（需重训校准） | **基本不受影响** |
| 落地成本 | 低 | 高（重提缓存+重训） | **低（即插即用）** |

风险提示：ratio 与 wmf 强负相关（-0.82），若与 mf3 条件衰减叠加可能双重惩罚多押谱 → 方案C 修正系数应只作用于 above_avg_density_mean（密度维度），不触碰 MF_FEATS。

---

## 10. 结论

1. **审计通过**：eff_ 计算逻辑正确（同押去冗余语义合理），问题不在 eff 本身，而在 above_avg_density_mean 未接入去冗余信息。
2. **问题确认**：上架谱 ratio 系统性低于官谱同段（55.9% 为负残差），高段 dens 虚高 5-11%；ratio 残差方向与 v11.1 预测偏差一致（虚高→偏高，扎实→偏低），是自制谱偏差的重要特征根源。
3. **推荐方案C**（eff_density_ratio 残差修正）：平滑、无 cap、即插即用、不伤官谱；方案B 作为长期根治方向（重训时改用 eff 版 above_avg）。
4. 已产出脚本：tools/exp_v112_density_ratio.py、tools/exp_v112_density_planB.py、tools/exp_v112_density_planB_sim.py、tools/exp_v112_density_planC.py、tools/exp_v112_density_ratio_ranked.py、tools/exp_v112_density_cases.py（中间结果 tools/_tmp_ratio_analysis.pkl、_tmp_planB_results.pkl、_tmp_planB_sim.pkl、_tmp_planC_results.pkl）。
