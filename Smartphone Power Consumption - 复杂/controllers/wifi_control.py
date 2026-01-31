from __future__ import annotations

from dataclasses import dataclass

from models.battery_model import (
    UsageInputs,
    WiFiControl,
    WiFiParams,
    WiFiState,
    clamp01,
    wifi_capacity_bps,
    wifi_expected_power_w,
)


@dataclass(frozen=True)
class WiFiCostWeights:
        """WiFi 速率控制的一步代价权重（离散时间实现）。

        连续时间概念性阶段代价：
            L = w_rate * (R_served - R_req)^2 + w_power * P_wifi + w_smooth * (dR_cmd/dt)^2

        实现要点：
        - deficit = max(0, R_req - R_served)（只惩罚“不足”，不惩罚“超额”）
        - P_wifi 采用 WiFi 期望功耗（与CTMC状态概率、发射功率状态等有关）
        - smooth 惩罚控制更新之间命令变化 ΔR_cmd，抑制抖动

        调参方向：
        - w_rate ↑：更重视吞吐/排队，倾向更高 R_cmd（耗电↑）
        - w_power ↑：更省电，倾向更低 R_cmd（可能积压队列）
        - w_smooth ↑：命令更平滑但响应更慢
        """

        w_rate: float = 1.0e-13  # 速率缺口惩罚（将 bps^2 缩放到 O(1)）
        w_power: float = 1.0  # 功耗惩罚（乘在 P_wifi[W] 上）
        w_smooth: float = 5.0e-13  # 平滑权重（惩罚 ΔR_cmd^2）


@dataclass
class WiFiOneStepOptimalRateController:
    """WiFi 数据速率的一步最优控制器（myopic MPC）。

    控制量：R_cmd_bps，从离散的 802.11-like 速率集合中选取。

    目标：
    - 吞吐“够用”：服务到达流量并在目标队列延迟内排空积压
    - 功耗更低：降低 WiFi 期望功耗

    备注：本实现里实际“served rate”会乘以 TX 状态概率 state.p5，
    用于近似 WiFi 并非时时都在发射。
    """

    params: WiFiParams
    weights: WiFiCostWeights = WiFiCostWeights()

    update_interval_s: float = 1.0  # 控制器刷新周期（秒）

    # include 0 for "no transmit"（0 表示不发射/不发送）
    rate_set_mbps: tuple[float, ...] = (
        0.0,
        1.0,
        2.0,
        5.5,
        6.0,
        9.0,
        11.0,
        12.0,
        18.0,
        24.0,
        36.0,
        48.0,
        54.0,
        72.2,
        86.7,
        144.4,
        173.3,
        300.0,
    )

    _last_update_t: float = -1e18
    _last_r_cmd_bps: float = 0.0

    def reset(self, r0_bps: float = 0.0) -> None:
        self._last_update_t = -1e18
        self._last_r_cmd_bps = max(0.0, min(self.params.r_max_bps, r0_bps))

    def _required_rate(self, u: UsageInputs, state: WiFiState) -> float:
        r_arr = max(0.0, u.wifi_rate_bps)
        drain = state.q_wifi_bits / max(1e-3, self.params.target_queue_delay_s)
        return min(self.params.r_max_bps, r_arr + drain)

    def __call__(
        self,
        t_s: float,
        u: UsageInputs,
        state: WiFiState,
        t_amb_c: float,
        t_cpu_c: float,
        t_cell_c: float,
        dt_s: float,
    ) -> WiFiControl:
        """计算当前时刻的 WiFi 速率控制输出（逐参数说明）。

        参数：
        - t_s：当前仿真时间（s）
        - u：使用/业务输入（到达速率 wifi_rate_bps 等）
        - state：WiFi 当前状态（队列 q_wifi_bits、状态机概率 p5 等）
        - t_amb_c：环境温度（℃），用于上层统一接口（此控制器自身不直接使用）
        - t_cpu_c：CPU 温度（℃），用于上层统一接口（此控制器自身不直接使用）
        - t_cell_c：蜂窝模块温度（℃），用于上层统一接口（此控制器自身不直接使用）
        - dt_s：仿真步长（s），用于上层统一接口（此控制器自身不直接使用）

        返回：
        - WiFiControl：包含 r_cmd_bps（bps）的控制命令
        """
        if (t_s - self._last_update_t) < self.update_interval_s:
            return WiFiControl(r_cmd_bps=self._last_r_cmd_bps)

        r_req = self._required_rate(u, state)
        cap = wifi_capacity_bps(self.params, state)

        best_r = self._last_r_cmd_bps
        best_j = float("inf")

        for r_mbps in self.rate_set_mbps:
            r_cmd = min(self.params.r_max_bps, max(0.0, r_mbps * 1e6))

            # Served outgoing throughput depends on being in TX state.
            r_phy = min(r_cmd, cap)
            r_served = max(0.0, state.p5) * r_phy

            deficit = max(0.0, r_req - r_served)
            p_wifi = wifi_expected_power_w(self.params, state, WiFiControl(r_cmd_bps=r_cmd))

            j = (
                self.weights.w_rate * (deficit**2)
                + self.weights.w_power * p_wifi
                + self.weights.w_smooth * ((r_cmd - self._last_r_cmd_bps) ** 2)
            )

            if j < best_j:
                best_j = j
                best_r = r_cmd

        self._last_r_cmd_bps = max(0.0, min(self.params.r_max_bps, best_r))
        self._last_update_t = t_s
        return WiFiControl(r_cmd_bps=self._last_r_cmd_bps)
