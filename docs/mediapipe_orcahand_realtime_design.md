# MediaPipe 到 OrcaHand 实时控制软件设计

## 目标

本软件使用 MediaPipe 识别用户手部 21 个关键点，推理出与 OrcaHand 对应的 17 个自由度，再通过安全控制链路驱动 OrcaHand 跟随用户手势运动。

核心原则是：视觉识别结果不能直接控制硬件。所有输出必须经过平滑、映射、状态机、安全限位、突变检查、单关节开关和紧急停止逻辑。

## 配置来源

运行时配置固定来自当前项目的 `config` 目录：

- `config/config.yaml`
- `config/calibration.yaml`

`config/config.yaml` 提供：

- `joint_ids`
- `joint_roms`
- `neutral_position`
- `joint_to_motor_map`
- `motor_ids`
- `max_current`
- `control_mode`

`config/calibration.yaml` 提供：

- `calibrated`
- `wrist_calibrated`
- `motor_limits`
- `joint_to_motor_ratios`

`--live` 模式启动前必须检查 17 个 joint 都有 ROM、motor 映射、motor limit 和非零 ratio。任意安全必要字段缺失时拒绝进入 live。`--dry-run` 可以继续运行，但必须显示警告。

## 总体架构

数据流如下：

```text
Camera
  -> MediaPipe HandTracker
  -> LandmarkSmoother
  -> HandKinematics
  -> JointMapper
  -> JointSmoother
  -> SafetyController
  -> RuntimeStateMachine
  -> OrcaController
  -> OrcaHand.set_joint_positions()
```

模块职责：

- `HandTracker`：复用现有 MediaPipe hand landmarker，输出 21 点关键点、handedness、置信度。
- `LandmarkSmoother`：对关键点做抖动平滑，减少视觉噪声。
- `HandKinematics`：从关键点计算人手控制量，包括手指弯曲和侧向张开。
- `JointMapper`：把人手控制量映射到 OrcaHand 的 17 个 joint。
- `JointSmoother`：对 17 个 joint 角度做二次平滑。
- `SafetyController`：执行 joint ROM、motor limit、安全 offset、单帧最大变化和异常突变检查。
- `RuntimeStateMachine`：管理 preview、armed、live、tracking_lost、fault 状态。
- `OrcaController`：只通过 Orca core 的 `OrcaHand.set_joint_positions(joint_dict)` 发命令，不直接写 Dynamixel。

## 运行模式和状态机

程序默认启动到 `preview` 状态，只识别手部并显示推理结果，不控制硬件。

状态定义：

- `preview`：只显示 MediaPipe 骨架、17 个推理角度、安全裁剪结果。
- `armed`：用户点击“开始映射”后进入。系统继续识别和计算，但还不启用机器手输出。
- `live`：用户点击“启用机器手”后进入。只有此状态允许向 OrcaHand 发送安全后的 joint 命令。
- `tracking_lost`：MediaPipe 丢手、置信度不足或连续异常帧时进入。保持上一帧安全姿态，长时间丢失后慢速回 safe neutral。
- `fault`：配置错误、motor limit 异常、OrcaHand 写入异常、紧急停止触发时进入。停止映射并调用 `disable_torque()`。

状态转换：

```text
preview --开始映射--> armed
armed --启用机器手--> live
live --丢手/异常--> tracking_lost
tracking_lost --恢复稳定识别--> armed 或 live
任意状态 --紧急停止/严重错误--> fault
fault --用户手动复位--> preview
```

## UI 和控制按钮

第一版可以使用 OpenCV 窗口加键盘控制，也可以后续升级为简单 GUI。必须提供这些控制：

- “开始映射”：从 `preview` 进入 `armed`。
- “启用机器手”：从 `armed` 进入 `live`。
- “捕捉自然张开姿态”：把当前手势作为人手 neutral 基准。
- “停止映射”：退出 live，回到 preview 或 armed。
- “紧急停止”：立即进入 fault，停止映射并 disable torque。

键盘快捷键：

- `m`：开始或关闭映射。
- `l`：启用或关闭 live。
- `n`：捕捉 natural neutral。
- `Space` 或 `Esc`：紧急停止。
- `q`：安全退出程序。

## MediaPipe 关键点

MediaPipe 手部点位按标准编号：

- `0`：wrist
- `1-4`：thumb
- `5-8`：index
- `9-12`：middle
- `13-16`：ring
- `17-20`：pinky

第一版只跟踪一只手。优先选择 handedness 与 `config/config.yaml` 里的 `type` 一致的手；如果无法可靠判断，则选择置信度最高且与上一帧 wrist 位置最近的手。

## 17 自由度映射

OrcaHand 的 17 个 joint 为：

```text
thumb_mcp, thumb_abd, thumb_pip, thumb_dip,
index_abd, index_mcp, index_pip,
middle_abd, middle_mcp, middle_pip,
ring_abd, ring_mcp, ring_pip,
pinky_abd, pinky_mcp, pinky_pip,
wrist
```

映射表：

| Orca joint | MediaPipe 来源 | 控制含义 |
| --- | --- | --- |
| `thumb_mcp` | 拇指整体轴相对手掌平面 | 大拇指整体靠近或远离手掌 |
| `thumb_abd` | 拇指轴相对食指方向的侧向角 | 大拇指靠近或远离食指 |
| `thumb_pip` | `2-3-4` 关节点弯曲 | 拇指中段前后弯曲 |
| `thumb_dip` | 拇指末端弯曲估计 | 拇指最上面关节 |
| `index_abd` | `5` 相对中指掌骨方向的横向偏移 | 食指左右张开 |
| `index_mcp` | `0-5-6` 或掌平面投影弯曲 | 食指根部前后 |
| `index_pip` | `5-6-8` 综合 PIP+DIP 弯曲 | 食指上部弯曲 |
| `middle_abd` | 中指相对手掌中心线 | 中指左右张开，默认接近 0 |
| `middle_mcp` | `0-9-10` | 中指根部前后 |
| `middle_pip` | `9-10-12` 综合弯曲 | 中指上部弯曲 |
| `ring_abd` | `13` 相对中指掌骨方向横向偏移 | 无名指左右张开 |
| `ring_mcp` | `0-13-14` | 无名指根部前后 |
| `ring_pip` | `13-14-16` 综合弯曲 | 无名指上部弯曲 |
| `pinky_abd` | `17` 相对中指掌骨方向横向偏移 | 小指左右张开 |
| `pinky_mcp` | `0-17-18` | 小指根部前后 |
| `pinky_pip` | `17-18-20` 综合弯曲 | 小指上部弯曲 |
| `wrist` | 第一版固定 safe neutral | 暂不跟随视觉控制 |

MediaPipe 只能提供 21 个视觉关键点，无法真实分离所有机械自由度。因此第一版目标是稳定、可校准的控制量，不追求医学级真实骨骼角度。

## Wrist 策略

第一版不主动控制 `wrist`。`wrist` 仍保留在 17 joint 数据结构中，也参与配置检查和安全检查，但输出保持在 safe neutral。

这样可以先把手指映射调稳，再单独评估手腕控制。手腕姿态受摄像头角度影响较大，直接启用会增加误动作风险。

## Neutral 捕捉和人手校准

增加“捕捉自然张开姿态”功能。

用户把手自然张开后按按钮，系统记录当前 MediaPipe 几何量作为 human neutral baseline。后续映射时，不使用绝对视觉角度，而使用相对 neutral 的变化量。

这可以修正：

- 不同用户手型差异。
- 摄像头角度差异。
- 手掌初始倾斜造成的 abd 偏移。
- mcp/pip 初始角度偏置。

捕捉 neutral 时要求：

- 识别到单只稳定手。
- 连续若干帧置信度合格。
- 关键点比例稳定。
- 没有突变或 tracking_lost。

## 抖动平滑

抖动处理分两层。

第一层是 landmark 平滑：

- 对 MediaPipe 输出的 21 个关键点做指数滑动平均或 One Euro Filter。
- 根据时间间隔 `dt` 自适应更新。
- 手丢失或突变时不更新滤波状态。

第二层是 joint 平滑：

- 对映射后的 17 个 joint 角度做指数滑动平均。
- 对不同 joint 可使用不同平滑系数。
- mcp/pip 可以比 abd 响应稍快；abd 更容易抖，应更平滑。

推荐默认：

```text
landmark_smoothing_alpha = 0.45
joint_smoothing_alpha = 0.35
abd_smoothing_alpha = 0.25
```

后续如果抖动仍明显，可以改用 One Euro Filter，因为它能在慢速时更稳、快速动作时更跟手。

## 置信度门控

不能只看 MediaPipe 是否检测到 hand，还要判断结果是否像一只稳定的手。

每帧必须通过以下检查：

- 21 个关键点完整。
- handedness 置信度或 tracking 置信度达到阈值。
- 手掌 bounding box 尺寸在合理范围。
- wrist 到 mcp 的距离比例稳定。
- 各手指骨段长度比例没有突然跳变。
- 当前 wrist 与上一帧 wrist 位移没有超过速度阈值。
- 关键点不能大面积重叠或折叠成不合理形状。

未通过时，该帧不能进入 JointMapper，也不能影响机器手。

## 突变拒识别

安全模块必须检测突然大幅快速变化，因为这通常说明 MediaPipe 把非手物体识别成手，或关键点跟踪跳到了错误位置。

突变检查分三层：

1. Landmark 层：关键点位置、手掌尺度、骨段比例突变。
2. Joint 层：推理出的 joint 角度短时间跳变。
3. Motor 层：预估 motor position 短时间跳变。

如果单帧变化超过阈值：

- 拒绝当前帧。
- 保持上一帧安全输出。
- 记录拒绝原因。

如果连续异常帧超过阈值：

- 从 `live` 进入 `tracking_lost`。
- 暂停机器手跟随。
- 长时间无法恢复时慢速回 safe neutral。

## 安全 offset

joint ROM 和 motor limits 必须使用同一份安全 offset 逻辑。

每个 joint 定义一个 `safety_offset_deg`。joint 侧直接使用角度 offset，motor 侧用同一个角度 offset 乘以该 joint 的 `joint_to_motor_ratio` 换算为 motor 弧度 offset。

公式：

```text
joint_safe_min = joint_rom_min + safety_offset_deg
joint_safe_max = joint_rom_max - safety_offset_deg

motor_offset_rad = safety_offset_deg * joint_to_motor_ratio
motor_safe_min = motor_limit_min + motor_offset_rad
motor_safe_max = motor_limit_max - motor_offset_rad
```

这样 joint 和 motor 使用同一份安全余量，不会出现角度留了 5 度、motor 却只留了另一个不一致余量的问题。

启动时必须检查：

- `joint_safe_min < joint_safe_max`
- `motor_safe_min < motor_safe_max`
- offset 不会把任何 joint 或 motor 的可用范围挤没。

推荐默认 offset：

```text
thumb_*: 5 deg
*_abd: 4 deg
*_mcp: 5 deg
*_pip: 5 deg
wrist: 5 deg
```

实际实现中这些值应写入 runtime safety 配置，允许后续按 joint 调整。

## 每帧最大变化限制

安全模块对每个 joint 限制单帧最大变化：

```text
delta = target - previous_safe
limited_delta = clamp(delta, -max_delta_deg_per_frame, +max_delta_deg_per_frame)
next_safe = previous_safe + limited_delta
```

推荐默认：

```text
thumb_*: 3 deg/frame
*_abd: 2 deg/frame
*_mcp: 4 deg/frame
*_pip: 5 deg/frame
wrist: 2 deg/frame
```

`wrist` 第一版固定 neutral，但仍保留限速配置。

## Motor limit 二次检查

虽然 `OrcaHand.set_joint_positions()` 会按 `joint_roms` clamp 并把 joint 转成 motor position，本软件仍必须在发送前做 motor 二次检查。

每帧对所有启用 joint 计算预估 motor position。预估方式必须和 Orca core 保持一致：

- 从 `joint_to_motor_map` 找到 motor。
- 用 `joint_to_motor_ratios` 把 joint 角度变化转换为 motor 弧度变化。
- 根据 joint 是否 inversion 选择方向。
- 使用 `motor_limits` 和统一换算后的 `motor_offset_rad` 判断是否越界。

如果目标 joint 会导致 motor 超过 safe motor limit：

- 先裁剪 joint 到可解释的安全边界。
- 如果裁剪后仍无法保证 motor 安全，拒绝当前帧并进入 `tracking_lost` 或 `fault`。

## 单关节开关和增益

每个 joint 都有独立控制配置：

```yaml
joint_controls:
  index_mcp:
    enabled: true
    gain: 1.0
  thumb_abd:
    enabled: false
    gain: 0.7
```

行为：

- `enabled: false` 时，该 joint 固定为 safe neutral，不跟随手势。
- `gain` 作用在人手控制量映射到 Orca 角度之前，用来缩小或放大动作幅度。
- gain 默认 `1.0`。
- 第一阶段调试建议只启用食指和中指，确认安全后再逐步启用拇指、无名指、小指。

## 丢手保护

如果 MediaPipe 丢手或置信度不足：

- 小于 `0.3s`：保持上一帧 safe output，不更新滤波器。
- 超过 `0.3s`：进入 `tracking_lost`，停止跟随。
- 超过 `2.0s`：缓慢回 safe neutral。

慢速回 neutral 也必须经过每帧最大变化限制和 motor limit 检查。

## 紧急停止

紧急停止是 live 模式必需功能。

触发方式：

- `Space`
- `Esc`
- UI Stop 按钮
- 严重异常自动触发

触发后：

- 状态进入 `fault`。
- 停止发送任何新的手势映射命令。
- 调用 `hand.disable_torque()`。
- 保留错误日志。
- 只有用户手动复位后才能重新进入 `preview`。

## 日志和回放

每次 live 运行保存日志，建议同时输出 CSV 和 JSON。

记录内容：

- timestamp
- runtime state
- MediaPipe handedness 和置信度
- 原始 landmark 几何量
- 原始 joint 角度
- 平滑后 joint 角度
- gain 后 joint 角度
- safety clamp 后 joint 角度
- 预估 motor position
- 是否被拒绝
- 拒绝原因
- 是否触发 tracking_lost 或 fault

用途：

- 调试某个 joint 抖动。
- 判断安全裁剪是否过于保守。
- 回放异常帧，确认是否是 MediaPipe 错识别。
- 逐步调 gain、offset、max_delta。

## Orca 控制接口

硬件控制只通过 Orca core 高级接口：

```python
hand.set_joint_positions(joint_dict)
```

禁止在主控制程序里直接调用底层 Dynamixel 写位置 API。这样可以复用 Orca core 已有的 joint 到 motor 转换、配置读取和硬件生命周期管理。

连接流程：

1. 创建 `OrcaHand(config_path="config/config.yaml")`。
2. `hand.connect()`。
3. `hand.init_joints(move_to_neutral=True)` 或更保守地显式设置 torque、control_mode、max_current。
4. 进入 `armed` 后等待用户启用 live。
5. `live` 中只发送 SafetyController 输出的 joint dict。
6. 退出或 fault 时调用 `disable_torque()` 和 `disconnect()`。

## 测试策略

第一版应包含不接硬件也能运行的测试。

单元测试：

- 读取 `config/config.yaml` 和 `config/calibration.yaml` 并验证 17 joint 完整性。
- 对每个 joint 生成 safe ROM 和 safe motor limits。
- 验证 joint offset 和 motor offset 使用同一个 `safety_offset_deg`。
- 验证超出 joint ROM 的目标会被裁剪。
- 验证超出 motor limits 的目标会被拒绝或裁剪。
- 验证单帧变化不会超过 `max_delta_deg_per_frame`。
- 验证突变帧会被拒绝。
- 验证 disabled joint 保持 safe neutral。
- 验证 wrist 第一版保持 safe neutral。

集成测试：

- 使用保存的 landmark 样例输入，跑完整 `landmark -> joint -> safety` 管线。
- dry-run 模式显示 17 joint 输出，不连接硬件。
- live 模式在配置不完整时拒绝启动。

人工测试顺序：

1. dry-run 只看 MediaPipe 识别。
2. 捕捉 neutral。
3. 只启用 `index_mcp/index_pip`。
4. 启用食指完整 DOF。
5. 启用中指。
6. 逐步启用无名指、小指。
7. 最后调试拇指。
8. wrist 保持关闭。

## 非目标

第一版不做：

- 多手同时控制。
- 远程控制。
- wrist 视觉跟随。
- 复杂 3D 手部物理模型。
- 绕过 Orca core 直接写 motor。
- 自动修改 `config/config.yaml` 或 `config/calibration.yaml`。

## 验收标准

设计完成后的实现应满足：

- 程序使用当前项目 `config/config.yaml` 和 `config/calibration.yaml` 作为安全配置来源。
- MediaPipe 输出不会直接控制硬件。
- 必须有 preview、armed、live、tracking_lost、fault 状态。
- 必须有开始映射、启用机器手、捕捉 neutral、紧急停止控制。
- 17 个 Orca joint 都有明确映射策略，其中 wrist 第一版固定 safe neutral。
- 每个 joint 都执行 joint ROM safe offset 检查。
- 每个 motor 都执行 motor safe limit 检查。
- joint 和 motor 使用同一个安全 offset，并通过 ratio 换算。
- 每帧 joint 变化不超过配置的 max delta。
- MediaPipe 抖动经过平滑处理。
- 突然大幅快速变化会拒绝当前帧。
- live 模式配置不完整时拒绝启动。
- 每个 joint 支持 enabled 和 gain。
- live 运行保存可回放日志。
- 紧急停止会停止映射并 disable torque。
