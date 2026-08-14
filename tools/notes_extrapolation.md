# t4: 外推段(≥17.7)社区趋势 vs 模型预测分析

> 分析脚本: tools/exp_extrapolation_analysis.py ｜ 原始输出: logs/exp_extrapolation_analysis.txt
> 背景: 官谱定数上限 17.6（Rrharil AT），自制谱 17.7+ 为模型外推段。用户要求外推段"大致保持社区定级趋势即可"。

## 1. 谱面清单（社区定数 ≥17.7，共 7 张）

| id | 名称 | level | 来源 | 社区定数 | 模型预测 | 偏差 | gb | boost | tags |
|---|---|---|---|---|---|---|---|---|---|
| 43988 | Waking Shadows (feat. Eili) | AT Lv.18 | 上架 | 18.50 | 18.33 | **-0.17** | 13.40 | 4.92 | CorsaceOpen2023 |
| 46403 | Crossmythos Rhapsodia | AT Lv.18 | 上架 | 18.10 | 17.71 | **-0.39** | 13.57 | 4.14 | CHUNITHM |
| 16593 | DEUX EX MĀXHINĀ | AT Lv.18 | 上架 | 18.00 | 18.48 | **+0.48** | 13.42 | 5.06 | 高难,SDVX |
| 24881 | Makear | ST Lv.18 | 上架 | 18.00 | 17.78 | **-0.22** | 13.83 | 3.95 | LTC,CHUNITHM |
| 46228 | Submerged City | AT Lv.18 | 上架 | 18.00 | 17.61 | **-0.39** | 13.51 | 4.11 | NCT |
| 24969 | WACCA ULTRA DREAM MEGAMIX | ST Lv.FINAL | 特殊 | 17.70 | 18.28 | **+0.58** | 13.36 | 4.92 | LTC |
| 7707 | Annihilation in F# Minor | EX Lv.17 | 上架 | 17.70 | 17.78 | **+0.08** | 13.58 | 4.19 | regular |

（严格"上架"为 6 张；第 7 张 WACCA 位于 charts.json 的"特殊"列表，但同样在 predictions.csv 中且 diff=17.70 达标，一并纳入。）

## 2. 排序一致性（Spearman）

- **7 张 17.7+**: Spearman = **0.2143**, Pearson = 0.2026，逆序对数 8/21 —— **排序一致性弱**
  - 社区升序: WACCA(17.70) ≈ Annihilation(17.70) → Makear/Submerged/DEUX(18.00) → Crossmythos(18.10) → Waking Shadows(18.50)
  - 模型升序: Submerged(17.61) → Crossmythos(17.71) → Annihilation(17.78) ≈ Makear(17.78) → WACCA(18.28) → Waking(18.33) → DEUX(18.48)
  - 主要乱序来源: ① Submerged City 被压到最低（社区 18.00 并列第 3）② DEUX EX 被顶到最高（社区 18.00 并列第 3，模型认为 18.48 第一）③ WACCA 模型排第 5（社区垫底）
- **放宽到 17.0+ 共 17 张**: Spearman = **0.6005**, Pearson = 0.6080 —— 趋势尚可，偏差主要来自个别极端异常

## 3. 偏差模式

- **平均偏差 -0.005（几乎无系统性方向）**，但 **MAE = 0.330** 较大 → 外推段偏差是"个别极端、方向随机"，非整体系统偏移
- **高估 3 张**（pred > diff + 0.05）:
  - WACCA +0.58：特殊列表 ST Lv.FINAL；boost=4.92 高。疑似超高峰值密度触发 boost 过量
  - DEUX EX MĀXHINĀ +0.48：boost=**5.06（7 张中最高）**，SDVX/高难标签，116.3s 短谱堆密度 → 符合"密度多指全押虚高"已知盲区（文档 30.11.6b）
  - Annihilation in F# Minor +0.08：基本持平，可接受
- **低估 4 张**（pred < diff - 0.05）:
  - Crossmythos Rhapsodia -0.39、Submerged City -0.39：AT Lv.18 长谱（Submerged 3909 notes/371s，Crossmythos 1970/156s），耐力型谱面，模型 boost 不足
  - Makear -0.22：ST Lv.18 自定义 level（映射 IN），LTC/CHUNITHM 标签
  - Waking Shadows -0.17：社区最高 18.50，模型 18.33 已接近，偏差相对小
- **规律**: 高估谱 = boost>4.9 的短时高密度谱；低估谱 = boost<4.2 的长时耐力/配置谱。偏差方向与 boost 高度相关，说明外推段偏差主要来自 boost 组件而非 gb（gb 全在 13.36~13.83 窄带内，几乎不区分 17.7+）

## 4. level/解析异常检查

- **3 张自定义 level**，全部被 level_key 映射为 **IN**（app.py 对未知 level 默认 IN）:
  - Makear `ST Lv.18`、WACCA `ST Lv.FINAL`、Annihilation `EX Lv.17`
- **ST/EX 映射为 IN 的影响**: ST/EX 谱的 level onehot 与 IN 相同 → 模型无法区分"自定义高难 level"与 IN。WACCA(+0.58) 与 POLYBIUS(ST Lv.BadEnd, 17.10→18.35, **+1.25 全场最极端**) 均受此影响
- 7 张 json 全部存在（json_exists=True），无解析缺失
- 官谱无 Lv.18；AT Lv.18 / ST Lv.18 / EX Lv.17 / ST Lv.FINAL 均为自制谱扩展难度段，**预测管线把它们当官谱难度体系外的常规谱处理**

## 5. 结论与建议

1. **外推段无系统偏差，但个别谱极端失真**：7 张平均偏差 -0.005、MAE 0.33；DEUX EX(+0.48)/WACCA(+0.58) 高估、Submerged City/Crossmythos(-0.39) 低估。用户"大致保持社区定级趋势"的要求在 17.0+ 全体（Spearman 0.60）基本满足，17.7+ 窄段排序弱但无方向性偏移
2. **修正抓手是 boost 而非 gb**：7 张 gb 全在 13.36~13.83（±0.2 内），boost 3.95~5.06（±1.1）主导了全部偏差 → 外推段修正应针对 boost 组件的 cap/系数，或对峰值密度类 boost 特征（DEUX EX 类短时堆料谱）加封顶
3. **自定义 level 需处理**：ST/EX/FINAL 映射 IN 使 WACCA、POLYBIUS 等特殊谱预测失真；建议对 ST Lv.18+ / EX Lv.17+ / Lv.FINAL 等自定义高难 level 单独映射（如并入 AT）或加 level 惩罚
4. **边界谱参考**：17.0~17.7 段也存在极端异常（POLYBIUS +1.25、Birds of Plague -0.62、Bathin -0.48），说明外推失真不是从 17.7 才开始的，而是 17.0+ 高难段整体存在的个例问题
5. **与 t2 联动**：17.7+ 全部为多指谱（t2 结论 4），其中 DEUX EX 属"多指堆料虚高"典型（应压低），Submerged City 属"长耐力低估"典型（应抬高）——与 t2 的"多指-双指方向相反修正"策略一致

## 6. 复现

```bash
C:\Python314\python.exe tools/exp_extrapolation_analysis.py
# 输出: logs/exp_extrapolation_analysis.txt
```
