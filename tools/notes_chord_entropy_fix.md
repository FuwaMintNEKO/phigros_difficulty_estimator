# chord_size_entropy 特征 Bug 分析与修复方案

> 任务 t1 | 特征工程研究员 fe-engineer | 2026-08
> 相关文档: docs/估算偏高问题审计与修复.md 30.11.5 节（记录该特征"几乎全为 0 或负值"异常）

## 一、Bug 现象

官谱 982 张上 `chord_size_entropy` 几乎全为 0 或负值（-0.00 ~ 0.93），
且它是 boost 特征（co=0.034），特征失效直接影响模型对"和弦复杂度"维度的刻画。

## 二、定位（feature_extractor.py）

计算位置: `extract_features` 内, 第 318~337 行:

```python
cs = simultaneous['chord_sizes']              # {2: n2, 3: n3, 4: n4, 5: n5}
total_sim_ev = simultaneous['event_count']    # 只统计 sz>1 的多押窗口

if total_sim_ev > 0:
    cs_probs = np.array([cs.get(k, 0) for k in [2, 3, 4, 5]]) / max(total_sim_ev, 1)
    cs_probs = cs_probs[cs_probs > 0]
    features['chord_size_entropy'] = float(-np.sum(cs_probs * np.log2(cs_probs + 1e-10)))
    features['chord_entropy_norm'] = float(features['chord_size_entropy'] / np.log2(4.0))
    mf_ratio_3p = (cs.get(3, 0) + cs.get(4, 0) + cs.get(5, 0)) / max(total_sim_ev, 1)
    features['chord_complexity'] = float(features['chord_size_entropy'] * mf_ratio_3p)
else:
    features['chord_size_entropy'] = 0.0
    ...
```

上游: `_compute_simultaneous_notes` (第 1663 行起) 的 `chord_sizes` 只统计
**多押窗口**（sz>=2），单押（sz=1）窗口完全不计入:
```python
chord_sizes = {2: 0, 3: 0, 4: 0, 5: 0}
...
if sz > 1:
    ...
    key = min(sz, 5)
    chord_sizes[key] = chord_sizes.get(key, 0) + 1
```

## 三、根因分析（两个叠加 bug）

### Bug A（负值的直接来源）: 熵公式 log 参数错误

标准香农熵:  `H = -Σ p·log2(p)`（p>0）。
代码写成 `-Σ p·log2(p + 1e-10)`：

- 当某个类别概率接近 1（例如纯双押谱 `cs_probs=[1.0]`）时，
  `log2(1.0 + 1e-10) ≈ +1.44e-10 > 0`，于是
  `H = -1.0 × (+1.44e-10) < 0` —— **出现负熵（-0.00）**。
- 加 `1e-10` 的意图是防止 `log2(0)`，但 `cs_probs[cs_probs > 0]` 已经过滤了 0，
  完全不需要加；加在 log 内部反而在 p→1 时引入正偏差。

### Bug B（几乎全为 0 的直接来源）: 概率分布不完整（漏掉单押）

`chord_sizes` 只有 {2,3,4,5} 四个桶，**单押（sz=1）事件被排除在分布之外**。
官谱绝大多数音符是单押，多押中又以双押为主：

- 低难谱（EZ/HD）几乎全单押 → `total_sim_ev` 很小甚至 0 → 熵 = 0
- 高难谱双押海 → `cs_probs ≈ [1.0]` → 熵 ≈ 0（或负 0.00）
- 只有 3+/4+/5+ 押丰富的少数谱才有非零熵（最大值 0.93）

即该熵衡量的是"**多押事件内部**的大小分布"，而非"全谱面和弦大小分布"，
对官谱几乎无区分度 —— 与 30.11.5 观察完全一致。

### 附带问题: 归一化分母

`chord_entropy_norm = entropy / log2(4.0)`，对应 4 个桶 {2,3,4,5}。
修复后分布扩为 5 个桶 {1,2,3,4,5}，最大熵 `log2(5) ≈ 2.322`，分母需同步改为 `log2(5)`。

## 四、修复方案

> 说明: 不直接修改 feature_extractor.py，以下为提交给实现阶段的补丁方案。
> 注意: `_compute_simultaneous_notes` 返回值被 `avg_chord_size` (410-412行) 等复用，
> 因此**不改 chord_sizes 字典本身**（避免改变 avg_chord_size 语义），
> 而是新增 `single_events` 字段。

### 4.1 上游: _compute_simultaneous_notes 增加单押计数

在函数末尾 return 前加一行即可（单押窗口数 = 总窗口数 - 多押窗口数）:

```python
return {
    'max': max_sim, 'avg': total_sim / max(event_count, 1),
    'event_count': event_count, 'multi_finger_events': mf,
    'chord_sizes': chord_sizes, 'pos_spreads': pos_spreads,
    'weighted_mf_score_total': sum(weighted_mf_scores),
    'weighted_mf_score_mean': float(np.mean(weighted_mf_scores)) if weighted_mf_scores else 0,
    'discrete_mf_ratio': discrete_mf_count / max(total_mf_events, 1),
    'total_mf_events': total_mf_events,
    'single_events': len(windows) - event_count,   # ← 新增: 单押窗口数
}
```

### 4.2 熵计算修复（extract_features 318-337 行替换）

```python
cs = simultaneous['chord_sizes']
total_sim_ev = simultaneous['event_count']
single_ev = simultaneous.get('single_events', 0)

# 完整和弦大小分布: 单押 + 2/3/4/5 押, 共 5 类
counts = np.array([single_ev, cs.get(2, 0), cs.get(3, 0), cs.get(4, 0), cs.get(5, 0)], dtype=float)
if counts.sum() > 0:
    cs_probs = counts / counts.sum()
    cs_probs = cs_probs[cs_probs > 0]
    # 标准香农熵: p>0 已过滤, log2 直接作用于 p, 不再 +1e-10
    ent = float(-np.sum(cs_probs * np.log2(cs_probs)))
    features['chord_size_entropy'] = ent
    # 5 类分布最大熵 log2(5)
    features['chord_entropy_norm'] = float(ent / np.log2(5.0))
    mf_ratio_3p = (cs.get(3, 0) + cs.get(4, 0) + cs.get(5, 0)) / max(total_sim_ev, 1)
    features['chord_complexity'] = float(ent * mf_ratio_3p)
else:
    features['chord_size_entropy'] = 0.0
    features['chord_entropy_norm'] = 0.0
    features['chord_complexity'] = 0.0
```

要点：
1. **标准熵公式**：`-Σ p·log2(p)`，p>0 已过滤，log 内不加平滑项 → 消除负熵
2. **分布含单押**：5 类分布（1/2/3/4/5 押），熵反映全谱和弦复杂度 → 消除"几乎全 0"
3. **归一化分母** log2(5)，保持 [0,1] 区间
4. `chord_complexity = entropy × mf_ratio_3p` 保留（多押占比加权，语义不变）

## 五、验证结果（全量官谱 982，脚本 tools/_verify_chord_entropy_fix.py）

详细数据: logs/exp_chord_entropy_fix.txt（含每张谱的 bug/fix 熵、单押/双押占比）

| 指标 | bug 版 | 修复版 |
|---|---|---|
| 负值个数 | **849 / 982** | **0** |
| 恰好 0 | 85 | 85（全单押谱，熵=0 正确） |
| 正值个数 | 48 | 897 |
| 数值范围 | -0.00 ~ 1.08 | 0 ~ 1.544 |
| 均值 | 0.0305 | 0.5256（归一化 0.2264） |
| **与定数 Spearman** | **-0.028（无区分度）** | **+0.539（显著正相关）** |

按定数分桶的修复熵均值（完美单调递增）:

| 定数段 | n | bug 均值 | fix 均值 | fix 归一化 |
|---|---|---|---|---|
| [0,8) EZ 段 | 329 | -0.0000 | 0.3523 | 0.1517 |
| [8,11) | 197 | -0.0000 | 0.4575 | 0.1971 |
| [11,13) | 114 | 0.0051 | 0.5705 | 0.2457 |
| [13,14.5) | 120 | 0.0116 | 0.6365 | 0.2741 |
| [14.5,16) | 151 | 0.0506 | 0.7017 | 0.3022 |
| [16,17) | 58 | 0.2027 | 0.8264 | 0.3559 |
| [17,18) | 13 | 0.6591 | 1.1384 | 0.4903 |

代表性样本:
- 修复熵最高: modulus IN 1.544 / QZKago AT 1.474 / KMoeVIP IN 1.454（多押丰富谱，符合直觉）
- 修复熵最低(非0): 均为 single≈1.00 的纯单押 EZ/HD 谱（熵≈0 正确）

结论: 修复后特征**全为非负、有区分度、随定数单调上升**，可用于模型。

> 注: 修复版与 bug 版同样有 85 张为 0，这些是全单押谱（single_ratio=1.0），熵=0 在语义上正确；
> bug 版的 85 张 0 则是"无多押事件"的兜底赋值，两者含义不同。

## 六、影响面与后续

- 修改仅影响 chord_size_entropy / chord_entropy_norm / chord_complexity 三个特征
- 新增 single_events 字段不影响其他消费者（avg_chord_size 等仍读 chord_sizes）
- 重训时需更新特征缓存（tools/build_feats_cache_v11.py），并重算该特征的 p95/p99
  （30.9.7 记录的 boost 分位数缺失问题一并处理）
