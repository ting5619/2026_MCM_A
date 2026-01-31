from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from models.battery_model import ScreenParams, ScreenState, UsageInputs, clamp01, screen_power_w


@dataclass(frozen=True)
class ScreenCostWeights:
    r"""Stage cost weights for screen brightness optimal control.

    We minimize (discrete-time approximation of a continuous-time objective):

    J = \int ( w_track (L_eff - L_req)^2 + w_power P_screen + w_smooth (dL_cmd/dt)^2 ) dt

    In code we use a sampled-time surrogate with a smoothness term on command changes.
    """

    # 参数含义/调参方向：
    # - w_track ↑：更重视亮度跟踪（更“亮”，功耗可能↑）
    # - w_power ↑：更重视省电（可能变暗/跟踪误差↑）
    # - w_smooth ↑：更平滑（减少亮度抖动，但响应更慢）
    w_track: float = 8.0  # 亮度跟踪权重（惩罚 (L_eff-L_req)^2）
    w_power: float = 1.2  # 功耗权重（乘在 P_screen[W] 上）
    w_smooth: float = 0.4  # 平滑权重（惩罚 ΔL_cmd^2）


@dataclass
class ScreenOneStepOptimalController:
    """Receding-horizon optimal control with N=1 (myopic MPC).

    Control: L_cmd(t) in [0,1]

    Dynamics (screen reduced-order):
      L_eff' = (L_cmd - L_eff)/tau
      T_s'   = (1/C)(P_screen - (T_s-T_env)/R + kappa(T_cpu - T_s))

    At each update we pick L_cmd that minimizes a one-step objective balancing:
      - brightness tracking to requested brightness L_req
      - low power (P_screen)
      - smoothness vs previous command

    This is computationally cheap and still fits an optimal-control framework.
    """

    screen: ScreenParams
    weights: ScreenCostWeights = ScreenCostWeights()

    # controller timing
    update_interval_s: float = 5.0  # 控制器刷新周期（秒）；越大越省算力但响应越慢

    # discrete optimization
    grid_points: int = 51  # 网格搜索候选点数（越大越精细但更慢）

    # internal state
    _last_update_t: float = -1e18
    _last_cmd: float = 0.0

    def reset(self, l_cmd0: float = 0.0) -> None:
        self._last_update_t = -1e18
        self._last_cmd = clamp01(l_cmd0)

    def _predict_next_state(
        self,
        u: UsageInputs,
        screen_state: ScreenState,
        l_cmd: float,
        t_env_c: float,
        t_cpu_c: float,
        dt_s: float,
    ) -> ScreenState:
        tau = max(1e-3, self.screen.tau_response_s)
        l_eff = screen_state.l_eff
        t_s = screen_state.t_s_c

        l_eff_next = clamp01(l_eff + dt_s * (l_cmd - l_eff) / tau)
        s_mid = ScreenState(t_s_c=t_s, q_pixel=screen_state.q_pixel, l_eff=l_eff_next)
        p_scr = screen_power_w(self.screen, u, s_mid)

        dT_dt = (
            p_scr
            - (t_s - t_env_c) / self.screen.r_th_c_per_w
            + self.screen.kappa_cpu_w_per_c * (t_cpu_c - t_s)
        ) / self.screen.c_th_j_per_c
        t_s_next = t_s + dT_dt * dt_s

        return ScreenState(t_s_c=t_s_next, q_pixel=screen_state.q_pixel, l_eff=l_eff_next)

    def _objective(
        self,
        u: UsageInputs,
        screen_state: ScreenState,
        l_cmd: float,
        l_req: float,
        t_env_c: float,
        t_cpu_c: float,
        dt_s: float,
    ) -> float:
        next_state = self._predict_next_state(u, screen_state, l_cmd, t_env_c, t_cpu_c, dt_s)
        p_scr_next = screen_power_w(self.screen, u, next_state)

        track = self.weights.w_track * (next_state.l_eff - l_req) ** 2
        power = self.weights.w_power * p_scr_next
        smooth = self.weights.w_smooth * (l_cmd - self._last_cmd) ** 2
        return track + power + smooth

    def __call__(
        self,
        t_s: float,
        u_req: UsageInputs,
        screen_state: ScreenState,
        t_env_c: float,
        t_cpu_c: float,
        dt_s: float,
    ) -> float:
        """计算当前时刻的屏幕亮度命令 L_cmd（逐参数说明）。

        参数：
        - t_s：当前仿真时间（s）
        - u_req：使用输入（包含 brightness 作为亮度需求，0~1）
        - screen_state：屏幕状态（有效亮度 l_eff、屏幕温度 t_s_c 等）
        - t_env_c：环境温度（℃），用于预测下一步屏幕温度
        - t_cpu_c：CPU 温度（℃），用于屏幕-CPU 热耦合的近似影响
        - dt_s：仿真步长（s），用于一步预测

        返回：
        - L_cmd：屏幕亮度命令（0~1）
        """
        if (t_s - self._last_update_t) < self.update_interval_s:
            return self._last_cmd

        l_req = clamp01(u_req.brightness)

        best_u = self._last_cmd
        best_j = float("inf")

        # Small grid search over admissible control bounds
        n = max(3, int(self.grid_points))
        for i in range(n):
            l_cmd = i / (n - 1)
            j = self._objective(u_req, screen_state, l_cmd, l_req, t_env_c, t_cpu_c, dt_s)
            if j < best_j:
                best_j = j
                best_u = l_cmd

        self._last_cmd = clamp01(best_u)
        self._last_update_t = t_s
        return self._last_cmd
