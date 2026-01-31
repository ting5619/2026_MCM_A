from __future__ import annotations

from dataclasses import dataclass

from models.battery_model import (
    BluetoothControl,
    BluetoothParams,
    BluetoothState,
    UsageInputs,
    ble_expected_power_w,
)


@dataclass(frozen=True)
class BluetoothCostWeights:
        """BLE（蓝牙）速率控制的代价权重（离散时间实现）。

        连续时间概念性阶段代价（便于理解）：
            L = w_rate * (R_served - R_req)^2 + w_power * P_BLE + w_smooth * (dR_cmd/dt)^2

        代码实现中：
        - 主要惩罚“服务速率不足”（deficit = max(0, R_req - R_served)）
        - 同时惩罚 BLE 期望功耗
        - 再加上命令平滑项（避免频繁、大幅度改变 R_cmd）

        调参经验：
        - 增大 w_rate：更重视吞吐/排队延迟，倾向更高 R_cmd（耗电↑）
        - 增大 w_power：更省电，倾向更低 R_cmd（可能导致队列积压）
        - 增大 w_smooth：命令变化更平滑，响应更“钝”，但更稳定
        """

        w_rate: float = 3.0e-13  # 速率缺口惩罚权重（把 bps^2 量级缩放到 O(1)）
        w_power: float = 1.0  # 功耗惩罚权重（乘在 P_BLE[W] 上）
        w_smooth: float = 8.0e-13  # 平滑权重（惩罚 ΔR_cmd^2；越大越不愿意改速率）


@dataclass
class BluetoothOneStepOptimalRateController:
    """BLE 平均数据速率的一步最优控制器（myopic MPC）。

    控制量：R_cmd_bps ∈ [0, r_max]。

    备注：BLE 本质是“连接事件驱动”（connection event）。本工程中将 R_cmd
    解释为“平均目标吞吐率”，再由被控对象把它映射为连接事件占空比/发包频率等。
    """

    params: BluetoothParams
    weights: BluetoothCostWeights = BluetoothCostWeights()

    update_interval_s: float = 1.0  # 控制器刷新周期（秒）；越大越省算力但响应越慢
    grid_points: int = 21  # 网格搜索候选点数；越大越精细但计算更慢

    _last_update_t: float = -1e18
    _last_r_cmd_bps: float = 0.0

    def reset(self, r0_bps: float = 0.0) -> None:
        self._last_update_t = -1e18
        self._last_r_cmd_bps = max(0.0, min(self.params.r_max_bps, r0_bps))

    def _required_rate(self, u: UsageInputs, state: BluetoothState) -> float:
        r_arr = max(0.0, u.bt_arrival_bps)
        drain = state.q_bits / max(1e-3, self.params.target_queue_delay_s)
        return min(self.params.r_max_bps, r_arr + drain)

    def __call__(
        self,
        t_s: float,
        u: UsageInputs,
        state: BluetoothState,
        t_amb_c: float,
        t_adj_c: float,
        dt_s: float,
    ) -> BluetoothControl:
        """计算当前时刻的 BLE 速率控制输出（逐参数说明）。

        参数：
        - t_s：当前仿真时间（s）
        - u：使用/业务输入（到达速率、是否开启 bt_on 等）
        - state：BLE 当前状态（队列长度、PER 等）
        - t_amb_c：环境温度（℃），用于功耗/温度相关模型（此控制器自身不直接使用）
        - t_adj_c：相邻基底/机身温度（℃），用于功耗/温度相关模型（此控制器自身不直接使用）
        - dt_s：仿真步长（s），用于上层统一接口（此控制器自身不直接使用）

        返回：
        - BluetoothControl：包含 r_cmd_bps（bps）的控制命令
        """
        if max(0.0, min(1.0, u.bt_on)) < 0.5:
            self._last_r_cmd_bps = 0.0
            self._last_update_t = t_s
            return BluetoothControl(r_cmd_bps=0.0)

        if (t_s - self._last_update_t) < self.update_interval_s:
            return BluetoothControl(r_cmd_bps=self._last_r_cmd_bps)

        r_req = self._required_rate(u, state)

        best_r = self._last_r_cmd_bps
        best_j = float("inf")

        n = max(5, int(self.grid_points))
        for i in range(n):
            r_cmd = self.params.r_max_bps * i / (n - 1)

            # Plant-side served rate depends on duty and PER; we use a cheap proxy:
            # r_served ≈ min(r_cmd, r_phy*(1-PER))
            r_cap = self.params.r_phy_bps * max(0.0, 1.0 - state.per)
            r_served = min(r_cmd, r_cap)

            deficit = max(0.0, r_req - r_served)
            p_ble = ble_expected_power_w(self.params, state, BluetoothControl(r_cmd_bps=r_cmd))

            j = (
                self.weights.w_rate * (deficit**2)
                + self.weights.w_power * p_ble
                + self.weights.w_smooth * ((r_cmd - self._last_r_cmd_bps) ** 2)
            )

            if j < best_j:
                best_j = j
                best_r = r_cmd

        self._last_r_cmd_bps = max(0.0, min(self.params.r_max_bps, best_r))
        self._last_update_t = t_s
        return BluetoothControl(r_cmd_bps=self._last_r_cmd_bps)
