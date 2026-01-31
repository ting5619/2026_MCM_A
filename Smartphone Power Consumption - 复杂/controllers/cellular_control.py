from __future__ import annotations

from dataclasses import dataclass

from models.battery_model import (
    CellularControl,
    CellularParams,
    CellularState,
    UsageInputs,
    cellular_capacity_bps,
    cellular_expected_power_w,
    clamp01,
)


@dataclass(frozen=True)
class CellularCostWeights:
        """蜂窝速率控制的一步代价权重（离散时间实现）。

        连续时间概念性阶段代价（用于理解/对照论文写法）：
            L = w_rate * (R_served - R_req)^2 + w_power * P_radio + w_smooth * (dR_cmd/dt)^2

        代码实现中：
        - 用 deficit = max(0, R_req - R_served) 作为“服务不足”的代价来源
        - P_radio 使用蜂窝模块的期望功耗（与状态机概率、发射功率状态等有关）
        - smooth 项惩罚相邻控制更新之间的命令变化 ΔR_cmd

        调参方向：
        - w_rate ↑：更偏向满足吞吐/排空队列（能耗↑，温度↑的风险↑）
        - w_power ↑：更偏向省电（可能积压队列，吞吐下降）
        - w_smooth ↑：命令更平滑（抑制抖动，但瞬态响应变慢）
        """

        w_rate: float = 2.0e-13  # 速率缺口惩罚（将 bps^2 量级缩放到 O(1)）
        w_power: float = 1.0  # 功耗惩罚权重（乘在 P_radio[W] 上）
        w_smooth: float = 1.0e-12  # 平滑权重（惩罚 ΔR_cmd^2）


@dataclass
class CellularOneStepOptimalRateController:
    """蜂窝数据速率的一步滚动优化控制器（myopic MPC）。

    控制量：R_cmd_bps ∈ [0, R_max]。

    目标：
    - “够用”：尽量服务到达流量并在目标延迟内排空积压队列
    - “省电”：降低蜂窝期望功耗 P_radio
    """

    params: CellularParams
    weights: CellularCostWeights = CellularCostWeights()

    update_interval_s: float = 2.0  # 控制器刷新周期（秒）；越大响应越慢但更省算力
    grid_points: int = 21  # 候选集合密度；越大越精细但计算更慢

    _last_update_t: float = -1e18
    _last_r_cmd_bps: float = 0.0

    def reset(self, r0_bps: float = 0.0) -> None:
        self._last_update_t = -1e18
        self._last_r_cmd_bps = max(0.0, min(self.params.r_max_bps, r0_bps))

    def _required_rate(self, u: UsageInputs, state: CellularState) -> float:
        # Heuristic: serve arrivals plus try to drain backlog within target delay.
        r_arr = max(0.0, u.cell_arrival_bps)
        drain = state.q_data_bits / max(1e-3, self.params.target_queue_delay_s)
        return min(self.params.r_max_bps, r_arr + drain)

    def __call__(
        self,
        t_s: float,
        u: UsageInputs,
        state: CellularState,
        t_amb_c: float,
        t_cpu_c: float,
        dt_s: float,
    ) -> CellularControl:
        """计算当前时刻的蜂窝速率控制输出（逐参数说明）。

        参数：
        - t_s：当前仿真时间（s）
        - u：使用/业务输入（到达速率 cell_arrival_bps 等）
        - state：蜂窝当前状态（队列 q_data_bits、状态机概率、发射功率状态等）
        - t_amb_c：环境温度（℃），用于蜂窝期望功耗模型
        - t_cpu_c：CPU 温度（℃），用于蜂窝功耗与热耦合的近似影响
        - dt_s：仿真步长（s），用于上层统一接口（此控制器自身不直接使用）

        返回：
        - CellularControl：包含 r_cmd_bps（bps）的控制命令
        """
        if (t_s - self._last_update_t) < self.update_interval_s:
            return CellularControl(r_cmd_bps=self._last_r_cmd_bps)

        r_req = self._required_rate(u, state)
        cap = cellular_capacity_bps(self.params, state)

        best_r = self._last_r_cmd_bps
        best_j = float("inf")

        # Candidate set: coarse global grid + dense neighborhood around r_req.
        candidates: list[float] = [0.0, r_req]

        # Dense local grid (helps when r_req is small relative to r_max)
        local_max = min(self.params.r_max_bps, max(1e6, 2.0 * r_req))
        local_n = max(9, int(self.grid_points))
        for i in range(local_n):
            candidates.append(local_max * i / (local_n - 1))

        # Coarse global grid to allow large rates when needed
        global_n = max(7, int(self.grid_points // 2))
        for i in range(global_n):
            candidates.append(self.params.r_max_bps * i / (global_n - 1))

        # De-duplicate and clamp
        uniq: list[float] = []
        for r in sorted(set(candidates)):
            if 0.0 <= r <= self.params.r_max_bps:
                uniq.append(r)

        for r_cmd in uniq:
            r_served = min(r_cmd, cap)

            deficit = max(0.0, r_req - r_served)
            p_radio = cellular_expected_power_w(
                self.params,
                state,
                CellularControl(r_cmd_bps=r_cmd),
                t_amb_c=t_amb_c,
                t_cpu_c=t_cpu_c,
            )

            j = (
                self.weights.w_rate * (deficit**2)
                + self.weights.w_power * p_radio
                + self.weights.w_smooth * ((r_cmd - self._last_r_cmd_bps) ** 2)
            )
            if j < best_j:
                best_j = j
                best_r = r_cmd

        self._last_r_cmd_bps = max(0.0, min(self.params.r_max_bps, best_r))
        self._last_update_t = t_s
        return CellularControl(r_cmd_bps=self._last_r_cmd_bps)
