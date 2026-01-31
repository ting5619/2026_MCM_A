# Smartphone Power Consumption — Continuous-Time SOC Model (简化版)

本工作区给出一套**连续时间**的锂离子电池 SOC 模型，并把 GPS / WiFi / 蜂窝 / CPU / 屏幕等复杂“状态机+微分方程”简化为**占空比/事件率驱动的平均功耗模型**，用于预测不同场景下的 *time-to-empty*。

## 1) 核心SOC控制方程（连续时间）

用 SOC \(z(t)\in[0,1]\) 表示剩余电量比例。

**电压近似（开路电压）**

\[
V_{oc}(z) = V_{min} + (V_{max}-V_{min})\,z
\]

**有效容量（温度+老化简化）**

\[
C_{eff}(T,\alpha)=C_{nom}\,(1-\alpha)\,g_T(T)
\]

其中 \(\alpha\in[0,1)\) 为容量衰减比例（老化），\(g_T(T)\) 为温度修正（见代码默认：在低温下线性下降，常温附近为1）。

**SOC微分方程（功率驱动）**

将系统总功耗记为 \(P_{sys}(t)\)（由各组件功耗相加得到），电源转换效率记为 \(\eta_{pmu}\)。电池电流近似
\[
I_{bat}(t) \approx \frac{P_{sys}(t)}{\eta_{pmu}\,V_{oc}(z(t))}
\]
SOC 的连续时间动力学：
\[
\frac{dz}{dt} = -\frac{I_{bat}(t)+I_{sd}}{C_{eff}(T(t),\alpha)}
\]
其中 \(I_{sd}\) 是自放电/漏电等等效电流（可设很小或置0）。

## 2) time-to-empty

给定初始 \(z(0)=z_0\) 与截止SOC \(z_{min}\)（如 0.02），定义
\[
T_{empty}=\inf\{t\ge0\mid z(t)\le z_{min}\}
\]
代码中用数值积分（Euler/RK）求解并返回耗尽时间。

## 3) 组件功耗降阶思想（把“变参”改为“定参/少参”）

### A. 通用模板：状态机 → 平均功耗

把多状态功耗写成：
\[
P_{comp}(t) \approx \sum_k D_k(t)\,P_k
\]
- \(P_k\)：状态k的**常数功耗**（或少量线性参数）
- \(D_k(t)\)：状态占空比（确定性调度）或马尔可夫链稳态概率

进一步把复杂连续状态（如温度、RSSI、C/N0）只保留对功耗影响最大的**一个一阶修正**，其余固定为常数。

### B. GPS（简化为“捕获事件 + 跟踪占空比”）

\[
P_{GPS}(t) \approx D_{fix}(t)\,P_{track} + \lambda_{acq}(t)\,E_{acq}
\]
- \(P_{track}\)：跟踪/解算平均功耗（常数）
- \(E_{acq}\)：每次冷启动/重新捕获的能量代价（J）
- \(\lambda_{acq}\)：每秒发生捕获事件的次数（可由“遮挡/室内”场景设定为常数）

### C. WiFi（简化为“空闲 + 吞吐量线性项 + 扫描事件”）

\[
P_{WiFi}(t) \approx P_{idle} + e_{bit}\,R_{bit}(t) + \lambda_{scan}(t)E_{scan}
\]
- \(R_{bit}\)：吞吐量（bit/s）
- \(e_{bit}\)：每比特能耗（J/bit，吸收TX/RX/MIMO等复杂性）

### D. 蜂窝（RRC tail 的一刀切近似）

\[
P_{cell}(t) \approx P_{idle} + D_{tx}(t)P_{high} + D_{tail}(t)P_{tail}
\]
其中 \(D_{tail}\) 可由“每次数据突发后持续 \(\tau_{tail}\)”得到。

### E. CPU（利用率线性+温度泄漏一阶修正）

\[
P_{cpu}(t) \approx P_{cpu,base} + k_u\,u_{cpu}(t) + P_{leak0}\,\exp(k_T (T(t)-25))
\]
在竞赛写作里这比 DVFS 全展开更好解释、更好估参。

### F. 屏幕（亮度主导：线性或幂律）

本仓库实现了一个**连续时间屏幕子模型**（见 `ScreenParams/ScreenState`），对齐你给出的方程结构：

**状态与输入**

- 状态：\(x=[T_s,\;Q_{pixel},\;L_{eff}]^T\)
- 输入：\(u=[L_{cmd},\;\gamma,\;f_r,\;R_{scale},\;A_{active}]^T\)

**功耗分解**
\[
P_{screen}=P_{backlight}+P_{driver}+P_{leakage}
\]

**OLED 亮度-功率（降阶后可估参）**
\[
P_{backlight}\approx P_{max}\,L_{eff}^{\gamma}\,\exp(\theta(T_s-25))\,A_{active}\,R_{scale}
\]

**驱动功耗（刷新率/分辨率/活跃面积）**
\[
P_{driver}\approx P_{drv,60}\,(f_r/60)\,A_{active}\,R_{scale}
\]

**温度动力学（热一阶网络+CPU耦合）**
\[
\dot T_s=\frac{1}{C_{th}}\left(P_{screen}-\frac{T_s-T_{env}}{R_{th}}+\kappa(T_{cpu}-T_s)\right)
\]

**亮度执行器动态**
\[
\dot L_{eff}=\frac{L_{cmd}-L_{eff}}{\tau_{resp}}
\]

**像素电荷代理状态（用于刻画刷新切换强度）**
\[
\dot Q_{pixel}=k_q\,f_r\,L_{eff}-\frac{Q_{pixel}}{\tau_q}
\]

（注：当前实现里 \(Q_{pixel}\) 主要用于后续扩展；你若希望把它显式进入 \(P_{driver}\) 或加入电荷相关损耗，我也可以继续接上。）

## 4) 快速开始

- 推荐（新简化温度耦合模型，SOC+系统温度两状态）：`python -m scripts.simulate_simplified --scenario piecewise_day --compare`
- 旧版（保留的详细模型/控制器/规划器入口）：`python -m scripts.simulate`
- 场景在 `scenario_lib/scenarios.py` 中配置（亮度、CPU利用率、网络吞吐等的分段常数）。

## 5) 屏幕亮度的最优控制（Optimal Control）

为满足“**足够亮**”与“**低功耗**”的矛盾目标，本仓库对屏幕亮度引入最优控制框架（见 `optimal_control.py`），控制量为亮度命令 \(L_{cmd}(t)\in[0,1]\)。

### 5.1 连续时间代价函数（可直接写进论文）

给定用户/应用的亮度需求 \(L_{req}(t)\)，定义运行代价：
\[
J=\int_0^{T}\Big( w_{track}(L_{eff}(t)-L_{req}(t))^2 + w_{power}\,P_{screen}(t) + w_{smooth}\,\dot L_{cmd}(t)^2 \Big)\,dt
\]

- 第一项：保证观感（\(L_{eff}\) 跟踪 \(L_{req}\)）
- 第二项：惩罚功耗（延长 time-to-empty）
- 第三项：限制频繁调光（提升体验、避免抖动）

### 5.2 控制器实现（滚动优化 / MPC 简化版）

实现采用“滚动时域优化”（Receding Horizon / MPC）的**一步预测**版本：

- 每隔 \(\Delta t_u\) 秒更新一次亮度
- 在 \([0,1]\) 上做小规模网格搜索，最小化一步近似的离散代价
- 输出最优 \(L_{cmd}\)，并由屏幕动力学 \(\dot L_{eff}=(L_{cmd}-L_{eff})/\tau\) 平滑执行

这保持了“最优控制”的结构（目标函数+约束+动力学），同时计算量足够低，适合竞赛建模与快速实验。

你可以在 `simulate.py` 里调 `ScreenCostWeights(w_track, w_power, w_smooth)` 观察不同权重下亮度与续航的折中。

## 6) 处理器 DVFS + 多核 的最优控制（Optimal Control）

本仓库把 CPU 从“经验线性项”升级为你给出的 **DVFS + 多核 + 热模型**的连续时间建模（见 `CpuParams/CpuState`），并实现一个控制器同时调节：

- 核心频率 \(f_c(t)\)
- 激活核心数 \(N_{active}(t)\)
- 利用率命令 \(u_{cmd}(t)\)（通过一阶动态影响 \(u_{eff}(t)\)）

### 6.1 连续时间动力学（降阶实现）

**电压-频率耦合（DVFS）**
\[
V_{dd}(f)=V_{min}+a\,[f-f_{min}]_{+}^{\beta}
\]

**功耗分解**
\[
P_{cpu}=P_{dyn}+P_{static}+P_{clock}
\]
\[
P_{dyn}=C_{eff}\,N_{active}\,f\,V_{dd}(f)^2\,u_{eff}
\]
\[
P_{static}=K_{leak}\,N_{active}\,V_{dd}(f)\,\exp\Big(\kappa\,(T_j/T_0-1)\Big)
\]
\[
P_{clock}=C_{clock}\,N_{active}\,f\,V_{dd}(f)^2
\]

**利用率一阶响应**
\[
\dot u_{eff}=\frac{u_{cmd}-u_{eff}}{\tau_{workload}}
\]

**热模型（一阶 RC 网络，带可选线性对流项）**
\[
C_{th}\,\dot T_j=P_{cpu}-\frac{T_j-T_{amb}}{R_{th}}-h(T_j-T_{amb})
\]

**热节流导致的频率上限**
\[
f_{max}(T_j)=f_{rated}\,[1-\gamma\,\max(0,T_j-T_{throttle})]
\]

### 6.2 算力指标与代价函数

场景给出归一化的算力需求 \(d(t)\in[0,1]\)（代码中为 `cpu_demand`），定义归一化供给：
\[
perf(t)=\frac{N_{active}}{N_{total}}\,\frac{f}{f_{rated}}\,u_{eff}
\]

最优控制目标（连续时间思想）：
\[
J=\int_0^T \Big( w_{perf}(perf-d)^2+w_{power}P_{cpu}+w_f\,\dot f^2+w_n\,\Delta N_{active}^2+w_u\,\dot u_{cmd}^2\Big)\,dt
\]

### 6.3 控制器实现（滚动优化 / MPC 简化版）

见 `cpu_control.py`：

- 每隔 \(\Delta t_u\) 秒更新一次
- 在候选网格上搜索 \(f\) 与 \(N_{active}\)
- 对每个候选 \((f,N_{active})\)，解析选取 \(u_{cmd}\approx \mathrm{sat}(d/((N/N_{tot})(f/f_{rated})))\)
- 用一步预测的功耗与算力误差组成代价，选最小者

这样既保留了“最优控制（目标函数+约束+动力学）”结构，也能在仿真里快速跑完并做敏感性分析。

如需绘图：安装 `numpy matplotlib`。

