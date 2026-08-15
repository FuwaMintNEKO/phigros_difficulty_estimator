# Parser 审计报告：unified_parser.py / predict_rpe.py

审计日期：本次会话
审计范围：`predict_rpe.py`（210 行）、`unified_parser.py`（437 行）全量代码
验证方式：通读代码 + 对 `data/test_charts/`、`data/chart/` 真实谱面样本（RPE 单线谱 epsilon/ex7/gamma、RPE 高仿 Apollo(17.8)、官方愚人节谱 Sigma (Haocore Mix)、PE 谱 RENDA JOCEKY/スタートリップ、官谱 AT 30+ 张）做结构/数值实测

## 结论速览

| 严重度 | 编号 | 位置 | 一句话 |
|--------|------|------|--------|
| 严重 | BUG-1 | predict_rpe.py:101-112 | RPE 判定线事件（move/rotate/alpha）完全不转换为官谱格式 |
| 严重 | BUG-4 | unified_parser.py:36-41,52-92 | 'rpe_v3' 分支永远不可达（死代码），RPE 愚人节谱处理失效 |
| 中 | BUG-5 | unified_parser.py:52-92 | _is_rpe_v3 检测条件与实际 RPE 结构不匹配 |
| 中 | BUG-7 | unified_parser.py:355,362 | PE 判定线 Y 坐标中心/缩放错误（300 应为 700） |
| 中 | BUG-8 | unified_parser.py:403-408 | PE cr 旋转角度未转弧度 |
| 中 | BUG-2 | predict_rpe.py:43-45 | BPMList 原样保留 [m,b,d]，未转标准 tick |
| 中 | BUG-3 | predict_rpe.py:15-22,105 | speedEvents 保留 RPE 格式（[m,b,d] 时间 + RPESpeed 单位）未转换 |
| 中 | BUG-6 | unified_parser.py:95-170 | _convert_rpe_v3_to_standard 按 positionX 分 bin、事件全丢（死代码内） |
| 低 | BUG-9 | unified_parser.py:340 | PE BPMList startTime 未 ×32 转 tick，与音符单位不一致 |

已核实**正确**（排除嫌疑）：RPE 类型映射 {1:1,2:3,3:4,4:2}（predict_rpe.py:9）；RPE startTime [m,b,d]→tick 数值公式（predict_rpe.py:71，m*4*8 恒等于 m*32，实测 [8,0,1]→256 ✓）；音符 positionX ÷75（predict_rpe.py:91，实测 -390→-5.2 ✓）；PE 音符 positionX 直接除（中心 0，实测 ±819.2 对称，unified_parser.py:383/388 ✓）。

---

## BUG-1【严重·核心】RPE 判定线事件不转换为官谱格式

- **位置**：predict_rpe.py 行 101-112（new_line 构造）
- **代码依据**：
  ```python
  new_line = {
      'bpm': base_bpm,
      'notesAbove': notes_above,
      'notesBelow': notes_below,
      'speedEvents': _merge_speed_events(line),
  }
  if line.get('eventLayers'):
      new_line['eventLayers'] = line['eventLayers']   # ← 只原样保留，不转换
  if line.get('extended'):
      new_line['extended'] = line['extended']
  ```
- **事实**（实测 Apollo(17.8).json，RPEVer=140）：RPE 判定线事件全部在 `eventLayers[0]` 下：moveXEvents×760、moveYEvents×657、rotateEvents×151、alphaEvents×241。moveX/Y 的 start/end 是**像素坐标**（视口 ±675/±450），rotate 是**角度（度）**，alpha 是 0-255，时间戳是 [m,b,d]。转换后这些只随 eventLayers 原样保留，**不生成**官谱的 judgeLineMoveEvents / judgeLineRotateEvents / judgeLineDisappearEvents。
- **影响**（实测量化）：
  1. **alphaEvents 完全丢失**：feature_extractor 只统计顶层 judgeLineDisappearEvents（feature_extractor.py:1365），RPE 谱 jline_disappear_density 恒为 0。实测 epsilon.json：转换后 jline_disappear_density=0.0；手动把 alphaEvents 转成 judgeLineDisappearEvents 后 = 1.586。
  2. **移动/旋转事件数值单位未转换**：像素(±675/±450)、角度(度) vs 官谱 positionX ±9 / positionY ±5 / 弧度；时间 [m,b,d] 未转 tick。任何读标准键名或直接用数值的下游全部失真（差 75/90 倍）。
  3. **与"高仿 move=0、差 100 倍"现象吻合**：官谱 AT 谱平均 judgeLineMoveEvents ≈ 3.4 万/谱（实测 30 张 AT 均值 33810），而 RPE 高仿版事件是 moveX+moveY 分开存储且数量少一个量级，加上 alpha 事件不计入消失密度，判定线类特征整体差 1~2 个数量级。
- **修复建议**：在 convert 时把 eventLayers 转换为官谱键（时间统一 (m+b/d)*32）：
  - moveXEvents + moveYEvents 按 startTime 合并 → judgeLineMoveEvents：`{'startTime':t0, 'endTime':t1, 'start':x0/75, 'end':x1/75, 'start2':y0/90, 'end2':y1/90}`（positionY 官谱范围 ±5，应 ÷90 而非 ÷75，见 BUG-7 佐证）
  - rotateEvents → judgeLineRotateEvents：`{'startTime':t0,'endTime':t1,'start':deg*pi/180,'end':deg*pi/180}`
  - alphaEvents → judgeLineDisappearEvents：`{'start':1 if alpha<128 else 0, ...}`
  - 转换后 eventLayers 可保留（feature_extractor 兼容计数），但必须同时生成标准键。

## BUG-4【严重·死代码】'rpe_v3' 分支永远不可达

- **位置**：unified_parser.py 行 36-41（detect_format）与 52-92（_is_rpe_v3）
- **代码依据**：
  ```python
  # detect_format 行 36-37
  if 'META' in data and 'RPEVersion' in data.get('META', {}):
      return 'rpe'              # ← key 存在即短路
  # _is_rpe_v3 行 72
  if not meta.get('RPEVersion'):   # ← 要求 truthy，与行 36 矛盾
      return False
  ```
- **事实**：任何带 `META.RPEVersion` 的谱（含 RPE 单线愚人节谱）在 detect_format 行 36 已被返回 'rpe'；而 _is_rpe_v3 又要求 RPEVersion 为真值。两条检查互为前置矛盾 → `_is_rpe_v3` 不可能返回 True → `_convert_rpe_v3_to_standard`（行 95-170）是**死代码**。
- **实测**：data/test_charts 中 RPE 单线谱 epsilon/ex7/ex8/gamma/ボーカルに無茶させんな/恋ひ恋ふ縁（RPEVer=113/140，jl=1，9703 notes）全部走 convert_rpe_to_standard（单线+事件不转换），从未触发 rpe_v3 分支。
- **影响**：RPE 愚人节单线谱不做任何特殊处理，与普通 RPE 谱一样单线化、事件不转换（叠加 BUG-1），谱面结构完全失真。
- **修复建议**：detect_format 先调 _is_rpe_v3 再判 'rpe'（或对 RPEVersion 为真值且单线+大物量的谱优先 rpe_v3），并同步修复 BUG-5 的检测条件。

## BUG-5【中】_is_rpe_v3 检测条件与实际 RPE 结构不匹配

- **位置**：unified_parser.py 行 68、79-90
- **代码依据**：
  ```python
  if 'numOfNotes' not in data:        # 行 68：查谱面顶层
      return False
  ...
  na = len(jl.get('notesAbove', []))  # 行 80-81：查 notesAbove/notesBelow
  ...
  has_events = any(k in jl for k in [  # 行 84-88：查官谱键名
      'judgeLineMoveEvents', 'judgeLineDisappearEvents', 'judgeLineRotateEvents'])
  ```
- **事实**（实测 epsilon.json / Apollo(17.8).json / 2155734445357448.json）：
  1. RPE 导出谱的 `numOfNotes` 在**判定线顶层**（如 `judgeLineList[0].numOfNotes = 9703`），谱面顶层没有该键 → 行 68 恒 False；
  2. RPE 音符在 `notes` 数组，没有 notesAbove/notesBelow → 行 80-81 恒得 n=0；
  3. RPE 事件在 `eventLayers[*].moveXEvents` 等，判定线顶层无 judgeLineMoveEvents 等官谱键 → 行 84-88 恒 False。
- **影响**：即使修复 BUG-4 的可达性，此函数也检测不到任何 RPE 谱（误判为普通 'rpe' 或 'standard'）。
- **修复建议**：改用 `jl.get('numOfNotes')` + `len(jl.get('notes', []))` + `'moveXEvents' in layer for layer in jl.get('eventLayers', [])` 组合判断。

## BUG-7【中】PE 判定线 Y 坐标中心与缩放错误

- **位置**：unified_parser.py 行 355（PEC_Y_SCALE）与 361-362（_move_ev）
- **代码依据**：
  ```python
  PEC_Y_SCALE = 700.0 / 9.0   # 行 355，注释称 "y中心300, 范围±700 → positionY±9"
  ...
  'start2': (y - 300.0) / PEC_Y_SCALE,   # 行 362
  ```
- **事实**（实测 RENDA JOCEKY.json，1716 个 cp/cm 事件）：PE y 坐标范围 **-7.78 ~ 1407.78，中心 700**（半范围 700）；官谱 judgeLineMoveEvents 的 start2/end2（positionY）范围 **±5**（实测 30 张官谱 AT absmax=5.0）。
- **错误**：① 中心 700 写成 300（偏移 400）；② 缩放基准 9 应为 5（y 半范围 700 → 官谱 ±5，正确 scale=140；代码 700/9≈77.78）。
- **影响**（实测）：PE 谱初始 cp `0 0 1024 700`（屏幕中心）→ start2=(700-300)/77.78=**5.14**（正确应为 0）。判定线 Y 位置整体偏移且缩放失真，线运动特征错误。
- **修复建议**：`'start2': (y - 700.0) / 140.0`（或 PEC_Y_SCALE = 700.0/5.0 = 140.0，中心改 700）。

## BUG-8【中】PE cr 旋转角度未转弧度

- **位置**：unified_parser.py 行 403-408
- **代码依据**：`'start': ang, 'end': ang`（ang 为角度）
- **事实**：官谱 judgeLineRotateEvents 用**弧度**（实测 30 张官谱 AT 非零值 0.138~6.0，1.569≈90°）；PE cr 角度范围 **-90 ~ 900（度）**（实测 RENDA JOCEKY 21 个 cr）。直接存角度当弧度，数值错 ~57 倍。
- **影响**：PE 谱旋转事件数值完全失真。
- **修复建议**：`ang * math.pi / 180.0`。

## BUG-2【中】BPMList 原样保留 [m,b,d]

- **位置**：predict_rpe.py 行 43-45
- **代码依据**：`data['BPMList'] = rpe_data.get('BPMList', [])`（startTime 仍是 [m,b,d] 数组）
- **事实**：RPE BPMList 形如 `{"bpm":350, "startTime":[308,13,768]}`（实测 epsilon/2155734445357448.json），官谱标准要求 startTime 为数字 tick（1拍=32tick）。当前 feature_extractor._parse_bpm_timeline 兼容 list 格式（feature_extractor.py:22-26）恰好掩盖，但任何按标准格式消费 BPMList 的路径（导出、对比、其他解析器）会失败或错位。
- **影响**：格式不标准；若下游按"数字"断言则抛错/忽略变速信息。
- **修复建议**：转换时生成 `{'startTime': (m + b/d) * 32, 'bpm': b}`。

## BUG-3【中】speedEvents 保留 RPE 格式未转换

- **位置**：predict_rpe.py 行 15-22（_merge_speed_events）、行 105
- **代码依据**：`events.extend(layer.get('speedEvents', []))` 后原样放入 new_line['speedEvents']
- **事实**（实测）：RPE speedEvents 形如 `{"start":13,"end":13,"startTime":[0,0,1],"endTime":[1,0,1]}`（RPESpeed 单位，1 RPESpeed = 120px/s；时间 [m,b,d]），官谱格式为 `{"startTime":tick,"endTime":tick,"value":倍率}`。feature_extractor 行 144-148 有 `value=start/5.0` 的兼容分支且未用事件时间，影响被部分掩盖。
- **影响**：格式不标准；事件时间保持 [m,b,d]，任何按 tick 消费的路径出错。
- **修复建议**：合并后统一转 `{'startTime':(m+b/d)*32,'endTime':…,'value':start/5.0}`。

## BUG-6【中·死代码内】_convert_rpe_v3_to_standard 策略错误

- **位置**：unified_parser.py 行 95-170（当前不可达，见 BUG-4）
- **代码依据**：行 158-161 按 `positionX` 分 bin 到 8 条虚拟线；虚拟线只带 speedEvents（行 154）；音符 positionX 未 ÷75（RPE 像素 ±675 直接保留，行 159）。
- **错误**：① RPE 单线愚人节谱的"多线"由判定线**时间维度的移动事件**产生，不是 positionX 空间分布，按 positionX 分 bin 完全错误；② 事件全丢（moveX/moveY/rotate/alpha 均不进入虚拟线）；③ 像素坐标未归一化到 ±9。
- **影响**：即使修复 BUG-4/5 激活此分支，输出仍是错误谱面。
- **修复建议**：正确的还原应按时间对每个音符求判定线移动后的实际 x 位置（从 moveXEvents 插值），再按时间窗聚类成虚拟线；或放弃虚拟多线化，改为保留单线 + 已转换的事件（见 BUG-1 修复）。

## BUG-9【低】PE BPMList startTime 未转 tick

- **位置**：unified_parser.py 行 340
- **代码依据**：`bpm_list.append({'startTime': float(parts[1]), 'bpm': bpm})`（bp 时间单位是拍，未 ×32）
- **对比**：同一文件音符时间 ×K=32（行 377）；官谱 BPMList startTime 为 tick。BPMList 与音符时间轴单位不一致。feature_extractor 对 float 当 beats（feature_extractor.py:28）碰巧兼容 PE 的拍单位，掩盖问题。
- **修复建议**：`'startTime': float(parts[1]) * K`。

---

## 已排除的嫌疑（供参考，不构成 bug）

1. **RPE startTime 公式语义**（predict_rpe.py:71）：`(m*4 + b*(4/d))*8` 把 m 当"小节"，但 m*4*8 恒等于 m*32，数值与官方 `(m+b/d)*32` 完全一致（实测 [8,0,1]→256、BPMList [308,13,768]→9856.54 均正确）。仅可读性差，非 bug。边界 d=0 时 try/except 静默降级为 0.0（行 72-73），RPE 合法谱 d 恒为正整数，低风险。
2. **PE 音符 positionX**（unified_parser.py:383/388）：实测 PE 音符 x 中心为 0（范围 ±819.2 对称），直接 `x/(1024/9)` 正确；判定线 x 中心 1024（范围 -7.59~2055.59 对称于 1024），`(x-1024)/scale` 正确。两坐标系各自处理正确。
3. **RPE 类型映射**（predict_rpe.py:9）：{1:1, 2:3, 3:4, 4:2} 与 PhiChartRender 权威映射一致。
4. **官方愚人节谱识别**：官方愚人节谱（Sigma (Haocore Mix) 等）为顶层 numOfNotes + 标准键（judgeLineMoveEvents 等）+ 无 RPEVersion → 被正确判为 'standard' 原样使用。

## 附加发现（两个文件之外，供参考）

- feature_extractor.py:28 `_parse_bpm_timeline` 把**标准 float startTime 直接当 beats**（官谱 BPMList startTime 是 tick），若官谱变速谱带数字 startTime 会错 32 倍（本地官谱样本 BPMList 多为空或 [m,b,d]，未触发）。
- RPE 音符 `speed` 字段（predict_rpe.py:88 保留原值）语义为 RPESpeed 倍率（RPE 判定线基准速度 1 RPESpeed=120px/s），与官谱音符 speed 倍率语义是否一致未见权威依据，未列入 bug。
