from __future__ import annotations

from dataclasses import dataclass

from models.battery_model import (
    GpsControl,
    GpsParams,
    GpsState,
    UsageInputs,
    gps_expected_power_w,
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _log2(x: float) -> float:
    import math

    return math.log(max(1e-12, x), 2)


@dataclass(frozen=True)
class GpsCostWeights:
        """GPS 一步最优控制的代价权重（离散时间实现）。

        控制器会在离散候选集合中选择：
        - f_update_hz：定位更新率（Hz）
        - sigma_pos_req_m：位置精度要求（m，越小越严格/越耗电）

        代价项包括：
        - 更新率不足：当 f_update_hz < UsageInputs.gps_update_min_hz 时产生惩罚
        - 精度不足：当预测的 sigma_pos_est_m > UsageInputs.gps_sigma_max_m 时产生惩罚
        - 期望功耗：gps_expected_power_w(...) 的输出（W）
        - 平滑项：抑制 f_update_hz 与 sigma_req 的频繁大幅跳变

        调参方向：
        - w_update_deficit ↑：更重视更新率（可能更耗电）
        - w_acc_deficit ↑：更重视精度（可能更耗电）
        - w_power ↑：更省电（可能牺牲更新率/精度）
        - w_smooth_* ↑：更平滑（响应更慢但更稳定）
        """

        w_update_deficit: float = 2.0  # 更新率不足惩罚权重
        w_acc_deficit: float = 8.0  # 精度不足惩罚权重（sigma_est 超过上限时）
        w_power: float = 1.0  # 功耗惩罚权重（乘在 P_gps[W] 上）
        w_smooth_f: float = 0.15  # 更新率命令平滑权重（惩罚 Δf^2）
        w_smooth_sigma: float = 0.05  # 精度要求平滑权重（惩罚 Δsigma^2）


@dataclass
class GpsOneStepOptimalController:
    """One-step myopic MPC controller for GPS update rate and accuracy requirement.

    Control variables:
      - f_update_hz in [0,10]
      - sigma_pos_req_m in [1,100]

    Requirements taken from UsageInputs:
      - gps_update_min_hz (minimum update rate)
      - gps_sigma_max_m  (maximum acceptable sigma_est)

    Other knobs are taken from UsageInputs unless overridden.
    """

    params: GpsParams
    weights: GpsCostWeights = GpsCostWeights()

    update_interval_s: float = 1.0  # 控制器刷新周期（秒）

    f_candidates_hz: tuple[float, ...] = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)  # 候选更新率（Hz）；0 表示关闭
    sigma_candidates_m: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 50.0, 80.0, 100.0)  # 候选精度要求（m）

    _last_update_t: float = -1e18
    _last_f_hz: float = 0.0
    _last_sigma_m: float = 50.0

    def reset(self, f0_hz: float = 0.0, sigma0_m: float = 50.0) -> None:
        self._last_update_t = -1e18
        self._last_f_hz = max(0.0, min(10.0, f0_hz))
        self._last_sigma_m = max(1.0, min(100.0, sigma0_m))

    def __call__(
        self,
        t_s: float,
        u: UsageInputs,
        state: GpsState,
        t_amb_c: float,
        t_substrate_c: float,
        dt_s: float,
    ) -> GpsControl:
        """计算当前时刻 GPS 的更新率/精度控制输出（逐参数说明）。

        参数：
        - t_s：当前仿真时间（s）
        - u：使用输入（gps_on、gps_update_min_hz、gps_sigma_max_m 等需求/约束）
        - state：GPS 当前状态（cn0_dbhz、sigma_pos_est_m、模式概率、温度等）
        - t_amb_c：环境温度（℃），用于上层统一接口（此控制器自身不直接使用）
        - t_substrate_c：基底/芯片邻接温度（℃），用于上层统一接口（此控制器自身不直接使用）
        - dt_s：仿真步长（s），用于一步预测 sigma_pos_est_m

        返回：
        - GpsControl：包含 f_update_hz（Hz）、sigma_pos_req_m（m）以及环路/辅助相关参数
        """
        # If app says GPS off and no legacy hint, force off.
        gps_enabled = 1.0 if max(0.0, min(1.0, u.gps_on)) >= 0.5 else 0.0
        if gps_enabled < 0.5 and (u.gps_fix_duty > 0.0 or u.gps_acq_rate_hz > 0.0):
            gps_enabled = 1.0

        if gps_enabled < 0.5:
            self._last_f_hz = 0.0
            self._last_sigma_m = max(1.0, min(100.0, u.gps_sigma_max_m))
            self._last_update_t = t_s
            return GpsControl(
                sigma_pos_req_m=self._last_sigma_m,
                f_update_hz=0.0,
                cn0_thresh_dbhz=u.gps_cn0_thresh_dbhz,
                b_loop_hz=u.gps_b_loop_hz,
                assist=u.gps_assist,
            )

        if (t_s - self._last_update_t) < self.update_interval_s:
            return GpsControl(
                sigma_pos_req_m=self._last_sigma_m,
                f_update_hz=self._last_f_hz,
                cn0_thresh_dbhz=u.gps_cn0_thresh_dbhz,
                b_loop_hz=u.gps_b_loop_hz,
                assist=u.gps_assist,
            )

        f_min = max(0.0, min(10.0, u.gps_update_min_hz))
        sigma_max = max(1.0, min(100.0, u.gps_sigma_max_m))

        best = (self._last_f_hz, self._last_sigma_m)
        best_j = float("inf")

        for f_hz in self.f_candidates_hz:
            f_hz = max(0.0, min(10.0, f_hz))
            for sigma_req in self.sigma_candidates_m:
                sigma_req = max(1.0, min(100.0, sigma_req))

                # One-step prediction of sigma_pos_est that captures the tradeoff:
                # tighter sigma_req improves estimation (at higher DSP power), and
                # higher f_update reduces between-update drift.
                import math

                n_locked = max(0.0, min(state.n_vis, state.n_locked))
                sat_gain = math.sqrt(max(1.0, n_locked))
                cn0_gain = _clamp((state.cn0_dbhz - self.params.cn0_min_dbhz) / 20.0, 0.2, 2.0)
                sigma_floor = self.params.sigma0_m / max(1e-6, sat_gain * cn0_gain)
                proc_gain = 1.0 + self.params.k_proc_gain * _log2(100.0 / sigma_req)
                sigma_floor_eff = sigma_floor / max(1.0, proc_gain)
                sigma_tcxo = self.params.k_sigma_tcxo_m * abs(state.t_gps_c - self.params.t_ref_c)

                update_term = self.params.k_sigma_update_m / max(0.1, f_hz) if f_hz > 1e-9 else 50.0
                sigma_target_nav = sigma_floor_eff + sigma_tcxo
                sigma_target_track = sigma_floor_eff + sigma_tcxo + update_term

                w_nav = _clamp(state.m4, 0.0, 1.0)
                tau_sigma = w_nav * self.params.tau_sigma_nav_s + (1.0 - w_nav) * self.params.tau_sigma_track_s
                sigma_target = w_nav * sigma_target_nav + (1.0 - w_nav) * sigma_target_track
                sigma_pred = state.sigma_pos_est_m + (-(state.sigma_pos_est_m - sigma_target) / max(1e-3, tau_sigma)) * dt_s

                update_def = max(0.0, f_min - f_hz)
                acc_def = max(0.0, sigma_pred - sigma_max)

                p_gps = gps_expected_power_w(
                    self.params,
                    state,
                    GpsControl(
                        sigma_pos_req_m=sigma_req,
                        f_update_hz=f_hz,
                        cn0_thresh_dbhz=u.gps_cn0_thresh_dbhz,
                        b_loop_hz=u.gps_b_loop_hz,
                        assist=u.gps_assist,
                    ),
                )

                j = (
                    self.weights.w_update_deficit * (update_def**2)
                    + self.weights.w_acc_deficit * (acc_def**2)
                    + self.weights.w_power * p_gps
                    + self.weights.w_smooth_f * ((f_hz - self._last_f_hz) ** 2)
                    + self.weights.w_smooth_sigma * ((sigma_req - self._last_sigma_m) ** 2)
                )

                if j < best_j:
                    best_j = j
                    best = (f_hz, sigma_req)

        self._last_f_hz, self._last_sigma_m = best
        self._last_update_t = t_s
        return GpsControl(
            sigma_pos_req_m=self._last_sigma_m,
            f_update_hz=self._last_f_hz,
            cn0_thresh_dbhz=u.gps_cn0_thresh_dbhz,
            b_loop_hz=u.gps_b_loop_hz,
            assist=u.gps_assist,
        )
