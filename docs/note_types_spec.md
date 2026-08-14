# Phigros Note 类型规范 (官方权威定义, 2026-08 用户实测确认)

> 本文件是 note 类型判定的唯一权威规范。改代码前必读。
> 来源: 用户实测 (Feeling Blue 全 hold 长按实证) + RPE/PE 格式分析 + 官谱行为对照

## 一、标准 Phigros 格式 type 定义

| type | 名称 | 判定 | 难度语义 | 核心密度 |
|---|---|---|---|---|
| 1 | tap (蓝键) | 点击 | 主要难度来源 | ✅ 计入 |
| 2 | drag (黄键) | 手指在上面即可判定 | 几乎零操作难度, 密集大量 | ❌ 不计 |
| 3 | hold (长条) | 长按 (holdTime) | 有难度 (耐力/长按) | ✅ 计入 |
| 4 | flick (红键) | 划动 | 比 tap 简单 (一只手划) | ❌ 不计 |

**核心密度** (core_mask = tap | hold) 只算 tap+hold:
- drag 不进核心密度 (零操作)
- flick 不进核心密度 (划动比点击简单)
- 速度类特征 (thirtysecond_run / fast_ms_* / interaction) 也只应统计 tap+hold 间隔 (v11.8c 待办: 排除 flick/drag 污染)

## 二、RPE 格式 type 映射 (predict_rpe.py RPE_TYPE_MAP)

```python
RPE_TYPE_MAP = {1: 1, 2: 3, 3: 4, 4: 2}  # v11.8c 修复
```

| RPE type | 判定依据 | 真实语义 | 映射到标准 |
|---|---|---|---|
| 1 | 瞬时 | Tap 蓝键 | 1 (tap) |
| 2 | 带 endTime (时长>0) | **Hold 长条 (长按!)** | **3 (hold)** |
| 3 | 瞬时 + 少位置 | Flick 红键 | 4 (flick) |
| 4 | 瞬时 + 多位置 | Drag 黄键 | 2 (drag) |

**铁证 (用户实测)**: Feeling Blue (47264) 游戏内全部为 Hold 长按 — 旧映射 {2:2,3:3,4:4} 把它解析成 Drag 导致完全失明 (预测 10.9 vs 修复后 14.4+)
**铁证 (格式)**: 3rd Avenue type=2 128 音符 endTime-startTime 全>0 (均值 3.36 拍) = 长条

## 三、PE 格式 type 映射 (unified_parser.py pe_type_map)

```python
pe_type_map = {'n1': 1, 'n2': 3, 'n3': 4, 'n4': 2}
```
| PE | 语义 | 标准 |
|---|---|---|
| n1 | Tap | 1 |
| n2 | Hold | 3 |
| n3 | Flick | 4 |
| n4 | Drag | 2 |

RPE 与 PE 同构 (n2=Hold→3, n3=Flick→4, n4=Drag→2) — PE 映射从未出错, v11.4 修复 RPE 时把正确的 {2:3,3:4,4:2} 改成了 {2:2,3:3,4:4}, 2026-08 用户实测发现并修复。

## 四、修复历史 (防止回退)

| 版本 | 映射 | 问题 |
|---|---|---|
| ≤v11.3 | {1:1, 2:3, 3:4, 4:2} | 正确 |
| v11.4 | {1:1, 2:2, 3:3, 4:4} | ❌ 错误! RPE Hold→Drag, Flick→Hold, Drag→Flick 全错位; 所有 RPE 谱特征错误; Feeling Blue 预测 10.9 (全 hold 当 drag 不计密度) |
| **v11.8c** | **{1:1, 2:3, 3:4, 4:2}** | ✅ 恢复正确 (用户实测发现) |

**RPE 修复的效果** (模型未重训, 仅类型修正后重预测):
- ranked MAE 0.568→0.551, bias -0.088→-0.021, rho 0.793→0.815, RMSE 0.780→0.724
- cyanine +2.06→-0.11 (flick/hold/drag 错位修复)
- Feeling Blue 10.92→14.44 (全 hold 正确计入)

## 五、注意事项
1. **官谱** (无 META.RPEVersion) 是标准格式, 不受映射影响
2. RPE 谱: notes 在 judgeLineList[].notes 或 notesAbove/notesBelow; 遍历要覆盖 all 三处
3. RPE 假音符 (isFake=1) 必须过滤 (装饰用)
4. drag 带 holdTime 是"跟随"不是"长按" — 但 RPE type2 的 endTime 表示 Hold (长按), 与 drag 的 holdTime 不同源
5. 修改映射后必须: 重建 feats_cache → 重预测 ranked/unranked (官谱无需重训, 模型不变)
