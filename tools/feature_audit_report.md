# feature_extractor.py 特征计算审计报告

- 审计对象: `feature_extractor.py`(当前版本 **v11.15c**,共 2020 行;行号均以该版本为准)
- 审计方法: 全文通读 + 合成谱数值验证 + 官谱实测(数据/chart 下 5 张官谱、RPE 原始谱 3 张)对照
- 已知事实(已核实): 1拍=32tick;16分=1/4拍(8tick)、24分=1/6拍(5.333tick)、32分=1/8拍(4tick)、48分=1/12拍(2.667tick)、64分=1/16拍(2tick);官谱音符 time 为整数 tick、positionX 实测范围约 ±7(RPE 转换后 ±9);官谱 holdTime 单位为 tick;官谱判定线 move 事件值为 ±0.5 屏幕坐标

---

## 一、严重(Bug 影响面大,数值差 32 倍或特征恒 0/恒 1)

### BUG-1 窗口密度体系:窗口参数用 tick,特征名/语义却是"拍"
- **行号**: 271-284(主窗口循环 `for w in [1,2,4,8,16]`)、289-291(t4)、296-301(core_std)、499-523(micro 0.0625/0.125/0.25)、542-576(耐力/手速 tap_d1)、739-763(小窗口 0.25/0.5/1)、766-777(burst d_half)；底层函数 `_compute_window_density` 在 1871-1878
- **bug**: `_compute_window_density(times, window)` 直接用 `window` 作为 **tick 宽度**建直方图(line 1875-1877: `bins=np.arange(0,(n_w+1)*window_size,window_size)`),而调用处传入的 w/mw 是"拍"数值(1,2,4,8,16 / 0.25,0.5,1 / 0.0625...),特征名全部带 `{w}beat`。1 拍 = 32 tick,窗口应为 w×32 tick,当前是 w tick
- **影响**: 约 40+ 个密度特征全部错位——`peak_density_1beat` 实为"1tick窗口"(≈最大同押数);`mean_density_1beat` 实为"每tick均值"(小 32 倍);`peak_density_4beat`、`tap_burst_*`、`core_std_density_*`、`micro_*`、`burst_*`、`density_above_zero_ratio`、`sustained_density_*`、`hand_speed_index` 的输入等全部基于错误窗口
- **验证**: 纯 16 分连打谱(1 拍 4 音符,窗口 1 拍应为 4)实测 `peak_density_1beat=1.0`、`peak_density_4beat=1.0`;DistortedFate IN(1 秒峰值 35)实测 `peak_density_1beat=4.0`(同押数,而非 1 拍密度)
- **修复**: 在 `_compute_window_density` 内将窗口换算为 tick(`window_size*32`),或调用处传 w×32;同时把 `mean_density` 等分母统一

### BUG-2 per-beat 密度除以 tick(小 32 倍)
- **行号**: 219(`notes_per_beat = n_notes / max(dt, 0.01)`)、221(tap_per_beat)、224(core_notes_per_beat)、541(tap_notes_per_beat);dt 定义于 179(`dt = float(times[-1])`,tick)
- **bug**: 特征名带 `beat` 却除以 tick 总数,单位是"每 tick",比"每拍"小 32 倍
- **验证**: 纯 16 分连打谱(100 音符/24.75 拍)实测 `notes_per_beat=0.126`(应 ≈4.04);DistortedFate 实测 0.1155(应 ≈3.7)
- **修复**: 除以 `dt/32`(即 duration_beats)

### BUG-3 hold 时长特征把 tick 当拍(大 32 倍)
- **行号**: 606(`total_hold_duration_beats = hold_time_sum`)、608(`avg_hold_duration_beats`)、609(`max_hold_duration_beats`)、614(`drag_hold_time_total_beats`)
- **bug**: holdTime 单位是 tick(1 拍=32 tick),这里直接赋给 `*_beats` 特征,数值是真实拍的 32 倍
- **验证**: 3 个 holdTime=32tick 的 hold,实测 `total_hold_duration_beats=96.0 / avg=32.0 / max=32.0`(正确应为 3.0/1.0/1.0);DistortedFate 实测 `avg_hold_duration_beats=17.35`(实际平均 ≈0.54 拍)
- **修复**: 除以 32(或 `time_to_seconds(...)*bpm/60`);注意 605/613 行的 `*_sec` 版本用恒定 bpm 估算,变速谱下也不准(次要)

### BUG-4 节奏间隔特征 tick 当拍 + 短间隔阈值用 tick
- **行号**: 637(`avg_interval_beats = np.mean(intervals)`)、638(std)、639(min);641(`short_interval_ratio` 阈值 0.25)、642(`very_short_interval_ratio` 阈值 0.125)
- **bug**: intervals 是 tick 差,赋给 `*_beats`(大 32 倍);0.25/0.125 意图是"拍"(=8/4 tick),却写成 tick 阈值,官谱整数 tick 下只有同押对(差 0)满足
- **验证**: 16 分连打谱 `avg_interval_beats=8.0`(应 0.25)、`short_interval_ratio=0.0`(应 ≈1);DistortedFate `avg_interval_beats=8.20`(应 ≈0.26)、`short_interval_ratio=0.185`(实为同押占比)
- **修复**: 间隔 ÷32;阈值 ×32(8/4)

### BUG-5 offbeat/weak 用 tick 的整数性判定拍位(官谱上恒 0 / 恒 1)
- **行号**: 1346-1348(offbeat)、1350(weak)
- **bug**: `offbeat = |times - round(times)| > 0.05`、`weak = |(times+0.5)%1 - 0.5| < 0.05`,判定的是 **tick 是否为整数**,而非"音符是否落在拍点/弱拍"。官谱 time 全是整数 tick → offbeat 恒 0、weak 恒 1
- **验证**: 三张官谱(DistortedFate/Igallta/GOODTEK)实测 `offbeat_ratio=0.0`、`weak_beat_ratio=1.0`,完全无信息
- **修复**: 先转拍 `bf = times/32`,offbeat = |bf-round(bf)|>0.05;弱拍按意图(如第 2/4 拍 |(bf-1)%2-1|<0.05 之类)重新定义

---

## 二、高(特征失效或数值明显错误,影响 2 个及以上特征)

### BUG-6 type_switch 交替阈值 0.5 tick(意图 0.5 拍)
- **行号**: 529-531(`times[i] - times[i-1] < 0.5 and nts[i] != nts[i-1]`)
- **bug**: 0.5 应为 0.5 拍(=16 tick)。官谱整数 tick 下仅"同 tick 的异种音符"(同押红蓝混合)满足,0.5 拍间隔的红蓝交替完全不计
- **验证**: 红蓝交替(间隔 16tick)谱实测 `type_switch_ratio=0.0`(应 ≈1.0);官谱仅 0.03~0.11(同押混合)
- **修复**: 阈值 ×32(16)或用秒域

### BUG-7 dense_mf 密集多押阈值 0.25 tick(意图 0.25 拍)
- **行号**: 407(`mf_times[i] - mf_times[i-1] <= 0.25`)
- **bug**: 3+ 押事件间隔 0.25 拍(=8 tick)应算"密集多押",0.25 tick 阈值几乎不触发
- **验证**: 3 押和弦连续(间隔 8tick)谱 `dense_mf_count=0`;三张官谱全部 `dense_mf_count=0`(特征恒 0 失效)
- **修复**: 阈值 ×32(8),并与 1637 行对拍特征(`<=16` = 0.5 拍)口径统一

### BUG-8 位置阈值 0.3/0.5 与 positionX 实际坐标范围不匹配(官谱/RPE 兼容)
- **行号**: 399-400(mf_cross_hand 的 has_left/has_right ±0.3)、421(cross_hand ±0.3)、1301-1303(left/right/center ±0.5)
- **bug**: 官谱音符 positionX 实测范围 **±7**(抽样 Igallta ±7.0、Spasmodic -7.56~7.64 等),RPE 转换后 ±9。±0.3/±0.5 阈值按"±1 归一化坐标"设计,量级差约 14~18 倍
- **影响**: `cross_hand_event_count` 虚高——DistortedFate 实测 212 事件(几乎任何 ≥2 音符同押窗口都被判交叉手);`mf_cross_hand` 同理;`left/right/center_ratio` 失去区分度(center 仅占 10~20%,且官谱 ±7 与 RPE ±9 刻度不一致)
- **验证**: 官谱 positionX 分布(见上);三张官谱 left≈right≈0.4、center≈0.1
- **修复**: 先归一化 positionX(官谱 ÷7、RPE ÷9,或统一 ÷9)再比较,或把阈值按实际刻度缩放(±0.3→≈±2.1 等);并统一官谱/RPE 刻度

### BUG-9 position_entropy 重复计算且坐标范围错误
- **行号**: 1448-1452(第一次,histogram range=(-2,2))、1688-1694(第二次,linspace(-1,1),覆盖前者)
- **bug**: 同一特征计算两次且范围不同(后者覆盖前者);官谱 ±7 音符大量被 clip 到边缘桶,熵严重失真
- **修复**: 删除一处;range 按谱面实际坐标范围(如 ±8)或归一化后计算

---

## 三、中(单个/少数特征失真或单位口径问题)

### BUG-10 per-tick 密度分母(与 BUG-2 同类)
- **行号**: 965(miniburst_density)、1262(trill_density)、1292(jack_density)、1440(wide_jump_density)、1328(rhythm_diversity)
- **bug**: 均除以 `dt`(tick),应为每拍(×32)或每秒;数值整体偏小且单位错误
- **修复**: 统一分母(duration_beats 或 duration_sec)

### BUG-11 其余 tick 域小阈值(意图拍/秒,官谱上基本只统计同押)
- **行号**: 1335(clutter `times[i]-times[i-1] < 0.04`)、1312(burst_moves `< 0.5`)、1439(wide_jump `time_gaps < 0.25`)、1634(chord_alternation `< 0.02`,注释写"0.01拍"=0.32tick)、1545(型切换和弦 `< 0.0625`,注释写"1/16拍"=2tick)
- **bug**: 均为"拍/秒"级阈值写成 tick 值,官谱整数 tick 下只有同押对(差 0)或极小浮点差满足,对应特征(`note_clutter_count`、`burst_avg_movement`、`wide_jump_count`、`chord_alternation_rate`、型切换"和弦"段识别)失效或语义错误
- **修复**: 按注释/意图换算成 tick(clutter 0.04s≈2.56tick@120、burst 0.5拍=16tick、wide 0.25拍=8tick、chord_alt 0.01拍=0.32tick、型切换 1/16拍=2tick)

### BUG-12 has_AT 用"是否有 flick"判断
- **行号**: 734(`features['has_AT'] = 1 if n_flick > 0 else 0`)
- **bug**: flick 音符存在与否与 AT 难度无必然关系(IN 谱同样大量 flick);若无难度标签,该特征几乎恒 1,无信息量且语义误导
- **修复**: 数据无难度标签时应删除该特征,或改用其他可解释代理

### BUG-13 above_below_cross 恒为 1.0
- **行号**: 1384-1385(`has_above_notes = has_above_notes or (na_raw is not None)` 等)
- **bug**: 官谱每条判定线都带 notesAbove/notesBelow 键(即使为空列表),`is not None` 恒真 → 特征恒 1.0,无区分度
- **验证**: 三张官谱全部 `above_below_cross=1.0`
- **修复**: 用非空判断(1375-1376 的真值判断即可),删除 is not None 覆盖

### BUG-14 chord_jack 分组未按判定线隔离
- **行号**: 1093-1108(分组逻辑 1090-1099)
- **bug**: 注释"同一判定线,时间间隔<4tick合并",实现只按时间切组(`times[i]-prev_t >= 4`),`g_lines` 只记录组内首个音符的线号。跨线同押(4tick 内多线音符)被并入同一组,后续"同线连续和弦重键"判定会混入跨线事件
- **修复**: 分组时同时校验判定线(不同线即使 4tick 内也另起一组)

### BUG-15 dt 取 times[-1] 而非首末差
- **行号**: 179(`dt = float(times[-1])`)
- **bug**: 官谱首音符通常不在 tick 0(如 DistortedFate 首音符 672tick),duration_beats/duration_sec/notes_per_second 等偏高约 5~10%
- **修复**: `dt = times[-1] - times[0]`(必要时加末音符半窗)

---

## 四、fast_ms_050_ratio 同线过滤(用户指定,仍存在)

### BUG-16 fast_ms_050/100/150 同线过滤丢失跨线密集
- **行号**: 999(`its = intervals_sec[same_line_mask & core_adj_full]`)、1000-1002
- **bug**: 只统计同线相邻间隔,跨线快速交替(Phigros 大量双线交错/对拍)完全不计。作者注释(995 行)认为"跨线交错非手指速度",但按用户实测底线,跨线密集同样反映手指速度/读谱压力,应计入
- **验证**: 双线交错谱(全局间隔 62.5ms,单线间隔 125ms)实测 `fast_ms_100_ratio=0.0`、`fast_ms_050_ratio=0.0`(而同谱 `global_jack_count=49` 能正确统计)
- **修复**: 移除 same_line 过滤(保留 core 与 >0 间隔过滤),或提供跨线版本特征

---

## 五、已确认修复(当前 v11.15c 不再存在,供回归参考)

1. `time_to_seconds` 积分模式被用于"间隔 tick"(返回累计时间而非间隔)——已改为局部 bpm 直算(657、955-956、1102 行)
2. fast_note_density 分音判定(旧: matched>=12 记 24th、>=16 记 48th 等级联错误)——已改为精确匹配 4/6/8/12/16(1726-1730 行)
3. RPE speedEvents 归一化(旧 /5)——已按官方规则 /4.5(149 行)
4. finger 特征窗口单位(旧 window_beats 公式错)——已改为 window_ticks=sec*32*bpm/60(1494 行)
5. RPE BPMList startTime [m,b,d] 的解析经数值验证与 predict_rpe 转换自洽(1 拍=32tick 一致),**不存在 ×4 错误**(曾怀疑,已排除)

---

## 六、验证摘要(实测数据)

| 合成谱 | 期望 | 实测(当前代码) |
|---|---|---|
| 16分连打 notes_per_beat | ≈4.04 | 0.126(小32倍) |
| 16分连打 peak_density_1beat | 4 | 1.0(1tick窗口) |
| 16分连打 avg_interval_beats | 0.25 | 8.0(tick当拍) |
| 64分连打 fast_64th | >0,其余0 | 仅64th>0(已修复) |
| 24分连打 fast_24th | >0,其余0 | 仅24th>0(已修复) |
| hold 32tick avg_hold_duration_beats | 1.0 | 32.0(tick当拍) |
| 红蓝交替(0.5拍) type_switch_ratio | ≈1.0 | 0.0(阈值0.5tick) |
| 3押连打(0.25拍) dense_mf_count | >0 | 0(阈值0.25tick) |
| 官谱×3 offbeat_ratio | 有分布 | 全 0.0 |
| 官谱×3 weak_beat_ratio | 有分布 | 全 1.0 |
| 官谱×3 above_below_cross | 有分布 | 全 1.0 |
