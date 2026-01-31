# Smartphone Power Consumption  简化温度耦合模型（SOC + 系统温度）

本项目已收敛为一套**极简但可解释**的功耗-温度-SOC 耦合模型：

- 状态变量只有 2 个：电池 SOC `z(t)` 与系统温度 `T_sys(t)`
- 温度会影响：电池有效容量、开路电压、屏幕/网络效率、CPU 泄漏功耗、GPS 精度
- 控制策略同时看 **SOC** 与 **温度**，按规则自动降亮度/降频/降速/关 GPS/降音量，并包含热保护与低温加热

核心实现见：
- `models/simplified_model.py`

## 1) 状态方程（连续时间）

### 1.1 系统温度动力学（简化热网络）

\[
\frac{dT_{sys}}{dt} = \frac{T_{amb} + \alpha_{heat} \cdot P_{heat} - T_{sys}}{\tau_T}
\]

- `P_heat` 使用原始组件功耗（不含效率修正、不含加热器），更贴近发热源的直觉。

### 1.2 电池温度（代数关系）

\[
T_{bat} = T_{sys} + \beta_{bat} \cdot P_{total}
\]

## 2) 温度对部件/电池的影响

### 2.1 电池容量温度修正

\[
 f_{T\_bat}(T_{bat}) =
 \begin{cases}
  1 - k_{cold}(T_{opt}-T_{bat})^2, & T_{bat}\le T_{opt}\\
  1 - k_{hot}(T_{bat}-T_{opt})^2, & T_{bat}> T_{opt}
 \end{cases}
\]

默认：`T_opt=25C, k_cold=0.005, k_hot=0.002`。

### 2.2 开路电压温度修正

\[
V_{oc} = [V_{min} + (V_{max}-V_{min})z] \cdot [1 + 0.003(T_{bat}-25)]
\]

### 2.3 屏幕效率温度修正

\[
\eta_{screen}(T_{sys}) = 0.92(1-0.002(T_{sys}-25)),\quad P_{screen,actual}=\frac{P_{screen}}{\eta_{screen}}
\]

### 2.4 CPU 泄漏功耗温度修正

\[
P_{cpu,leak}=P_{cpu,base}\exp(0.03(T_{sys}-25)),\quad P_{cpu,total}=P_{cpu,dyn}+P_{cpu,leak}
\]

### 2.5 网络效率温度修正

\[
\eta_{net}(T_{sys}) = 0.88(1-0.0015(T_{sys}-25)),\quad P_{net,actual}=\frac{P_{net}}{\eta_{net}}
\]

### 2.6 GPS 精度温度修正

\[
\sigma_{GPS}(T_{sys}) = \sigma_{base}[1+0.01|T_{sys}-25|]
\]

## 3) SOC 方程（带温度修正）

\[
\frac{dz}{dt} = -\frac{P_{total}(t)}{C_{nom}\,V_{oc}\,f_{T\_bat}(T_{bat})}
\]

代码中 `C_nom` 用 Ah，因此实现时额外除以 `3600` 把小时换成秒。

## 4) 温度感知控制策略

总控制系数：
\[
ctrl = \min(f_{SOC}(SOC), f_T(T_{sys}))
\]

- `f_SOC(SOC)=0.2+0.8SOC`
- `f_T(T_sys)` 按你的分段规则实现（20~30 内较温和，过冷/过热更强降额）

各部件控制：
- 屏幕：`L = L_req * min(ctrl, f_T_brightness)`（高温>35 降亮度）
- CPU：`u = u_req * min(ctrl, f_T_cpu)`（高温>40 降频/降负载）
- 网络：`R = R_req * min(ctrl, f_T_network)`（高温>45 降速）
- GPS：`T_sys>50` 直接关闭更新，否则按 `ctrl` 缩放
- 扬声器：`V = V_req * min(ctrl, f_T_speaker)`（高温>40 降音量）

热管理：
- `T_sys > T_throttle(55)`：额外把 `ctrl` 再乘 `0.5`
- `T_sys < T_heat(0)` 且 `SOC>0.3`：开启加热 `P_heater=k_heater(T_opt-T_sys)` 并计入 `P_total`
- `T_sys >= T_shutdown(65)`：视为强制关机（实现中将 SOC 置 0 结束仿真）

## 5) 运行

推荐命令：

- `python -m scripts.simulate --scenario piecewise_day --compare`

可选参数：
- `--t-amb` 环境温度（）
- `--t0` 初始系统温度（）
- `--z0` 初始 SOC
- `--t-max` 最长仿真时间（秒）
- `--no-control` 关闭温度/SOC 控制
- `--print-params` 打印默认参数

场景配置见：
- `scenario_lib/scenarios.py`

## 6) 代码结构

- `models/types.py`：`UsageInputs` 与 `clamp01`
- `models/simplified_model.py`：简化温度耦合模型 + 仿真 `simulate()`
- `scenario_lib/scenarios.py`：典型工作负载场景
- `scripts/simulate.py`：命令行入口
