# 尾杀窗口细化: tail_peak_5pct 与 tail_peak_last3s

> 任务 t3 | 特征工程研究员 fe-engineer | 2026-08
> 相关文档: docs/估算偏高问题审计与修复.md 30.11.5 节
> （"尾杀特征方向正确但窗口过宽：DF 最后5秒被 15% 窗口稀释"）

## 一、背景

现有 `compute_tail_features`（feature_extractor.py 第 1593 行起）用**末段 15% 时长**窗口：
`cut = total_sec * 0.85`。社区证据（30.11.5）表明：

- **DistortedFate AT 17.4**：难点集中在"最后 5 秒"超短尾杀 → 15% 窗口稀释
  （tail_peak_ratio 仅 0.90 不突出）
- **QZKago AT 17.4**：尾杀 20 秒，15% 窗口能正确识别（tail_core_share=0.27 全场最高）

目标：新增两个更细的尾杀窗口特征，与现有 15% 窗口并存：
1. `tail_peak_5pct`：末段 **5% 时长**窗口的 1s 峰值密度 / 全局 1s 均值密度
2. `tail_peak_last3s`：固定**末 3 秒**窗口的 1s 峰值密度 / 全局 1s 均值密度

## 二、设计

### 核心工具: 1s 滑动窗口峰值

现有代码用 1s 直方图 bin 近似峰值（`int((t-cut)/win)`），窗口短时粒度不足
（末 3 秒只有 3 个 bin）。改为真正的 **1s 滑动窗口（0.25s 步进）**:

```python
def _peak_1s_in_window(tsec_core, start, end, win=1.0, step=0.25):
    """区间 [start, end) 内 1s 滑动窗口峰值核心音符数（步进 step 秒）"""
    seg = tsec_core[(tsec_core >= start) & (tsec_core < end)]
    if seg.size == 0:
        return 0.0
    max_c = 0
    cur = start
    while cur + win <= end + 1e-9:
        c = int(((seg >= cur) & (seg < cur + win)).sum())
        if c > max_c:
            max_c = c
        cur += step
    return float(max_c)
```

### 扩展 compute_tail_features（新增部分）

```python
# ===== 全局基准 =====
tsec_core = tsec[core]
global_peak_1s = _peak_1s_in_window(tsec_core, 0.0, total_sec)
global_mean_1s = float(tsec_core.size / max(total_sec, 0.01))

# ===== 关键设计: 窗口锚定最后一个核心音符（不是总时长） =====
# 初版按总时长定窗口 (cut5 = total_sec*0.95, cut3 = total_sec-3) 验证发现:
# Phigros 官谱末尾常有静默 (最后音符早于总时长, 如 DF 152.4s vs 157.0s, Rrharil 125.1s vs 130.1s),
# 按总时长算的窗口落在空窗里 -> 特征为 0, 完全漏掉尾杀。
# 因此 5%/3s 窗口统一锚定最后一个核心音符 (last_note = tsec_core.max()):
#   - 5% 窗口: [last_note - total_sec*0.05, last_note]
#   - 3s 窗口: [last_note - 3.0, last_note]
# 15% 窗口保持原语义向后兼容。
# ===== 现有 15% 窗口（保留）=====
cut15 = total_sec * 0.85
p15 = _peak_1s_in_window(tsec_core, cut15, total_sec)
out['tail_peak_vs_mean'] = p15 / max(global_mean_1s, 0.01)
out['tail_peak_1s_ratio'] = p15 / max(global_peak_1s, 1.0)
out['tail_density'] = float(tsec_core[tsec_core >= cut15].size / max(total_sec - cut15, 0.01))
out['tail_core_share'] = float(tsec_core[tsec_core >= cut15].size / max(tsec_core.size, 1))

# ===== 新增: 5% 窗口（锚定最后音符） =====
last_note = float(tsec_core.max())
w5 = total_sec * 0.05
cut5 = max(last_note - w5, 0.0)
p5 = _peak_1s_in_window(tsec_core, cut5, last_note)
out['tail_peak_5pct'] = p5 / max(global_mean_1s, 0.01)          # 5%窗口1s峰值 / 全局均值
out['tail_peak_5pct_1s_ratio'] = p5 / max(global_peak_1s, 1.0)  # 5%窗口1s峰值 / 全局峰值
out['tail_density_5pct'] = float(tsec_core[tsec_core >= cut5].size / max(last_note - cut5, 0.01))

# ===== 新增: 固定末3秒（锚定最后音符） =====
cut3 = max(last_note - 3.0, 0.0)
p3 = _peak_1s_in_window(tsec_core, cut3, last_note)
out['tail_peak_last3s'] = p3 / max(global_mean_1s, 0.01)
out['tail_peak_last3s_1s_ratio'] = p3 / max(global_peak_1s, 1.0)
out['tail_density_last3s'] = float(tsec_core[tsec_core >= cut3].size / max(last_note - cut3, 0.01))
```

### 语义说明

| 特征 | 窗口 | 分子 | 分母 | 含义 |
|---|---|---|---|---|
| tail_peak_vs_mean (现有) | 末15% | 1s滑动峰值 | 全局1s均值 | 末段爆发/平均 |
| **tail_peak_5pct** (新) | 末5% | 1s滑动峰值 | 全局1s均值 | 超短尾杀爆发/平均 |
| **tail_peak_last3s** (新) | 固定末3s | 1s滑动峰值 | 全局1s均值 | 固定3秒爆发/平均 |

- 三特征并存：15% 管"尾杀段整体"，5% 管"超短爆发"，3s 管"最末端收尾"
- `tail_density_5pct` / `tail_density_last3s` 为辅助密度特征
- 保留现有特征名不变，新增键不破坏旧模型/旧缓存

## 三、验证（DistortedFate / QZKago / Rrharil 官谱 AT）

脚本 tools/_verify_tail_window.py，全量数据 logs/exp_tail_window.txt（981 张官谱）

### 3.1 重点谱对比（AT 难度）

| 谱面 (定数) | 15%窗口 tail_peak_vs_mean | **5%窗口 tail_peak_5pct** | **末3秒 tail_peak_last3s** | 末3秒 density | 社区评价 |
|---|---|---|---|---|---|
| DistortedFate (17.4) | 2.81 | **2.95** ↑ | **1.68** ↓ | 5.3 | "最后5秒全游最难尾杀"（爆发型短尾杀） |
| QZKago (17.4) | 3.50 | 3.50 | **3.27** | 22.7 | 尾杀20秒持续高压（持续型） |
| Rrharil (17.6) | 2.45 | 2.34 | 2.34 | 18.0 | 难点分散，尾杀也强 |

**判读（新特征成功区分"爆发型 vs 持续型"尾杀）**:

1. **DF 的 tail_peak_5pct (2.95) > 15% 窗口 (2.81)**：5% 窗口捕捉到比 15% 窗口
   更高的 1s 峰值（21 vs 20）——证实"超短尾杀被 15% 窗口稀释"（30.11.5 假设成立）。
   而 DF 的 tail_peak_last3s (1.68) 显著低于 5% 窗口——其尾杀爆发集中在
   "倒数 3~8 秒"（峰值 @~149s，最后音符 152.4s），最末 3 秒反而缓和。
2. **QZKago 末3秒 tail_peak_last3s=3.27、density=22.7 全场最高**：尾杀持续高压
   到最后 1 秒（峰值 @~131s，最后音符 134.2s）——与社区"尾杀20秒拉高一大档"一致。
3. **Rrharil 居中**：5% 与 3s 窗口接近（2.34/2.34），尾杀强度分布均匀。

→ **两个新特征 + 窗口差异 (5pct vs 3s) 能刻画"尾杀爆发时刻"**：
DF 型（爆发在倒数几秒）vs QZKago 型（持续到尾），这是 15% 窗口无法区分的。

### 3.2 全量官谱分布（981 张）

| 特征 | range | mean | Spearman~diff |
|---|---|---|---|
| tail_peak_vs_mean (15%) | 0~12.7 | 2.87 | -0.322 |
| **tail_peak_5pct** | 0~11.9 | 2.36 | -0.212 |
| **tail_peak_last3s** | 0~10.4 | 1.99 | -0.097 |
| tail_peak_1s_ratio (15%) | 0~1.33 | 0.85 | +0.075 |
| tail_peak_5pct_1s_ratio | 0~1.25 | 0.70 | +0.124 |
| tail_peak_last3s_1s_ratio | 0~1.25 | 0.60 | +0.168 |

16+ 段分桶（社区高难段）: 17-18 段 tail_peak_5pct 均值 2.70 > 16-17 段 1.95，
5% 窗口特征在最高难段区分度更强。

注: vs_mean 类特征与定数弱负相关、_1s_ratio 类弱正相关，与文档 30.9 结论
（"尾杀特征对全局贡献小、但对特定谱修正有效"）一致——尾杀特征的价值在
**修正 DF/QZKago 这类尾杀集中谱的偏差**，而非全局单调。

## 四、影响面与后续

- 新增 6 个特征键，现有 15% 窗口特征语义不变
- 修改 compute_tail_features 的峰值计算从直方图改为滑动窗口，
  会略微改变现有 tail_peak_vs_mean / tail_peak_1s_ratio 的数值（更精确），需重训
- 重训时更新特征缓存 tools/build_feats_cache_v11.py
