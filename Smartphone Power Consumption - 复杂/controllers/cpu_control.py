from __future__ import annotations

from dataclasses import dataclass

from models.battery_model import CpuControl, CpuParams, CpuState, UsageInputs, clamp01, cpu_perf, cpu_step, cpu_total_power_w, f_max_throttled


@dataclass(frozen=True)
class CpuCostWeights:
    r"""CPU 一步最优控制的代价权重（离散时间实现）。

    概念性目标（连续时间形式，便于理解）：
      J = \int [ w_perf (perf - demand)^2 + w_power P_cpu
                + w_smooth_f Δf^2 + w_smooth_n Δn^2 + w_smooth_u Δu^2 ] dt

    其中：
    - perf：归一化算力供给（与频率、活跃核数、利用率相关）
    - demand：应用需求（UsageInputs.cpu_demand，范围 0~1）
    - P_cpu：CPU 功耗（W）
    - Δf/Δn/Δu：与上一次控制输出相比的变化量（平滑/防抖）

    调参方向：
    - w_perf ↑：更追求性能满足（频率/核心数更积极，耗电↑、发热↑）
    - w_power ↑：更省电（可能性能不足）
    - w_smooth_* ↑：更平滑（响应更慢，但更稳定）
    """

    w_perf: float = 25.0  # 性能误差权重（perf 与 demand 的偏差惩罚）
    w_power: float = 0.8  # 功耗权重（乘在 P_cpu[W] 上）
    w_smooth_f: float = 0.8  # 频率变化平滑权重（惩罚 Δf^2）
    w_smooth_n: float = 0.3  # 核数变化平滑权重（惩罚 Δn^2）
    w_smooth_u: float = 0.2  # 利用率命令变化平滑权重（惩罚 Δu^2）


@dataclass
class CpuOneStepOptimalController:
    """One-step receding-horizon controller (myopic MPC) for CPU.

    Controls: (f_ghz, n_active, u_cmd)

    Practical simplification:
    - We treat workload demand as required normalized compute in [0,1]
    - For each (f, n) candidate, we pick u_cmd analytically to meet demand as much as possible
    - Then evaluate one-step predicted cost
    """

    cpu: CpuParams
    weights: CpuCostWeights = CpuCostWeights()

    update_interval_s: float = 1.0  # 控制器刷新周期（秒）；越大越省算力但响应越慢

    # Search grids（网格搜索密度）
    f_grid: int = 10  # 频率候选点数（越大越精细但更慢）
    n_grid: int = 9  # 核数候选点数（越大越精细但更慢）

    _last_update_t: float = -1e18
    _last_ctrl: CpuControl = CpuControl(f_ghz=1.0, n_active=1, u_cmd=0.0)

    def reset(self, ctrl0: CpuControl | None = None) -> None:
        if ctrl0 is None:
            ctrl0 = CpuControl(f_ghz=0.7 * self.cpu.f_rated_ghz, n_active=max(1, self.cpu.n_total // 2), u_cmd=0.0)
        self._last_ctrl = ctrl0
        self._last_update_t = -1e18

    def _pick_u_cmd(self, demand: float, f: float, n: int) -> float:
        denom = (n / self.cpu.n_total) * (f / self.cpu.f_rated_ghz)
        if denom <= 1e-9:
            return 0.0
        return clamp01(demand / denom)

    def _objective(self, t_s: float, u_req: UsageInputs, state: CpuState, ctrl: CpuControl, t_amb_c: float, dt_s: float) -> float:
        # One-step prediction
        next_state = cpu_step(self.cpu, state, ctrl, t_amb_c=t_amb_c, dt_s=dt_s)
        p_cpu = cpu_total_power_w(self.cpu, next_state, ctrl)
        perf = cpu_perf(self.cpu, next_state, ctrl)

        demand = clamp01(u_req.cpu_demand)
        perf_err = self.weights.w_perf * (perf - demand) ** 2
        power = self.weights.w_power * p_cpu

        df = (ctrl.f_ghz - self._last_ctrl.f_ghz)
        dn = (ctrl.n_active - self._last_ctrl.n_active)
        du = (ctrl.u_cmd - self._last_ctrl.u_cmd)
        smooth = self.weights.w_smooth_f * (df**2) + self.weights.w_smooth_n * (dn**2) + self.weights.w_smooth_u * (du**2)

        return perf_err + power + smooth

    def __call__(self, t_s: float, u_req: UsageInputs, state: CpuState, t_amb_c: float, dt_s: float) -> CpuControl:
        """计算当前时刻 CPU 的 DVFS/核心数/利用率控制输出（逐参数说明）。

        参数：
        - t_s：当前仿真时间（s）
        - u_req：使用输入（包含 cpu_demand，范围 0~1）
        - state：CPU 当前状态（结温 t_j_c、有效利用率等）
        - t_amb_c：环境温度（℃），用于 CPU 一步预测（cpu_step）
        - dt_s：仿真步长（s），用于 CPU 一步预测（cpu_step）

        返回：
        - CpuControl：包含 f_ghz（GHz）、n_active（个）、u_cmd（0~1）
        """
        if (t_s - self._last_update_t) < self.update_interval_s:
            return self._last_ctrl

        # Temperature-dependent frequency cap
        f_cap = max(self.cpu.f_min_ghz, f_max_throttled(self.cpu, state.t_j_c))
        f_min = self.cpu.f_min_ghz
        f_max = min(self.cpu.f_rated_ghz, f_cap)

        # Candidate grids
        f_count = max(3, int(self.f_grid))
        n_count = max(2, int(self.n_grid))

        best = self._last_ctrl
        best_j = float("inf")

        for i in range(f_count):
            f = f_min + (f_max - f_min) * i / (f_count - 1)
            for j in range(n_count):
                n = int(round((self.cpu.n_total) * j / (n_count - 1)))
                n = max(0, min(self.cpu.n_total, n))

                u_cmd = self._pick_u_cmd(clamp01(u_req.cpu_demand), f, max(1, n)) if n > 0 else 0.0
                ctrl = CpuControl(f_ghz=f, n_active=n, u_cmd=u_cmd)

                J = self._objective(t_s, u_req, state, ctrl, t_amb_c=t_amb_c, dt_s=dt_s)
                if J < best_j:
                    best_j = J
                    best = ctrl

        self._last_ctrl = best
        self._last_update_t = t_s
        return best
