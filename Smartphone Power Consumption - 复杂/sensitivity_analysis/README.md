# 灵敏度分析（数值扰动法）

本目录提供一个“基于物理仿真 + 数值扰动”的局部灵敏度分析工具。

## 目标
- 目标函数：电池耗尽时间 `t_empty_s`（`simulate_soc` 的返回值）
- 对每个被控量 `u_i`（例如 `l_cmd`、`cpu_f_ghz`、`wifi_r_cmd_bps` 等）做微小扰动，量化对 `t_empty_s` 的影响。

## 方法
- 基准运行：用既定控制策略（默认：one-step 控制器 + 热管理控制器）运行一次仿真，得到 `t0 = t_empty_s` 与 traces。
- 单变量扰动：只对一个控制器输出做微小缩放/偏移（其余不变），再次运行仿真得到 `t1`。
- 灵敏度（有限差分近似）：
  - 绝对灵敏度：`S = (t1 - t0) / Δu`
  - 弹性系数（归一化对比）：`E = (Δt/t0) / (Δu/ū)`，其中 `ū` 是该控制量在基准运行的时间平均值。

## 用法
在项目根目录运行：

- 打印表格（默认场景 + 默认扰动）：
  - `python -m sensitivity_analysis.run_sensitivity --scenario worst_case_nav_hotspot`

- 指定扰动幅度（相对扰动 eps，例如 0.01=+1%）：
  - `python -m sensitivity_analysis.run_sensitivity --scenario worst_case_nav_hotspot --eps 0.01`

- 写出 CSV：
  - `python -m sensitivity_analysis.run_sensitivity --scenario worst_case_nav_hotspot --out sensitivity.csv`

## 备注
- 本工具默认分析“控制器输出（命令）”对 `t_empty_s` 的灵敏度，即在既定策略附近的局部敏感性。
- 若要分析更复杂的策略（如 planner 的离散模式切换），建议扩展为“策略参数扰动”（例如 mode 的 `power_mult/track_mult`）或做全局采样回归。
