from __future__ import annotations

from dataclasses import dataclass

from models.battery_model import (
    SpeakerControl,
    SpeakerParams,
    SpeakerState,
    UsageInputs,
    speaker_expected_power_w,
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


@dataclass(frozen=True)
class SpeakerCostWeights:
        """扬声器增益控制的一步代价权重。

        代价项：
        - 响度不足：loud_req - loud_est（只惩罚不足，不惩罚超额）
        - 功耗：speaker_expected_power_w(...) 返回的电功耗（W）
        - 平滑：抑制增益 g 的频繁/大幅变化

        备注：loudness 在这里是“代理指标”（由音圈焦耳热/功率映射得到的 0~1 量），
        用于把音频体验约束融入到优化里。

        调参方向：
        - w_loud_deficit ↑：更重视音量/响度满足（更耗电/更热风险）
        - w_power ↑：更省电（可能音量不足）
        - w_smooth ↑：更平滑（响应更慢，减少抖动）
        """

        w_loud_deficit: float = 8.0  # 响度缺口惩罚权重
        w_power: float = 1.0  # 功耗惩罚权重（乘在 P_spk[W] 上）
        w_smooth: float = 0.6  # 平滑权重（惩罚 Δg^2）


@dataclass
class SpeakerOneStepOptimalVolumeController:
    """One-step myopic MPC controller for speaker gain G in [0,1]."""

    params: SpeakerParams
    weights: SpeakerCostWeights = SpeakerCostWeights()

    update_interval_s: float = 1.0  # 控制器刷新周期（秒）
    grid_points: int = 21  # g 的网格搜索点数（越大越精细但更慢）

    _last_update_t: float = -1e18
    _last_g: float = 0.0

    def reset(self, g0: float = 0.0) -> None:
        self._last_update_t = -1e18
        self._last_g = _clamp(g0, 0.0, 1.0)

    def __call__(
        self,
        t_s: float,
        u: UsageInputs,
        state: SpeakerState,
        t_amb_c: float,
        t_phone_c: float,
        t_cpu_c: float,
        v_bat_v: float,
    ) -> SpeakerControl:
        """计算当前时刻扬声器增益控制输出（逐参数说明）。

        参数：
        - t_s：当前仿真时间（s）
        - u：使用输入（speaker_volume 0~1、spk_audio_level 等）
        - state：扬声器状态（音圈温度、电流、滤波器电压等）
        - t_amb_c：环境温度（℃），用于上层统一接口（此控制器自身不直接使用）
        - t_phone_c：机身/外壳温度（℃），用于上层统一接口（此控制器自身不直接使用）
        - t_cpu_c：CPU 温度（℃），用于上层统一接口（此控制器自身不直接使用）
        - v_bat_v：电池端电压（V），用于计算功放/音圈功耗

        返回：
        - SpeakerControl：包含 g（0~1增益）、v_limit_v（V限幅）、f_mode（工作模式）
        """
        # If no audio content, shut down gain.
        audio_level = _clamp(getattr(u, "spk_audio_level", 0.0), 0.0, 1.0)
        vol_req = _clamp(u.speaker_volume, 0.0, 1.0)
        if audio_level <= 1e-4 or vol_req <= 1e-4:
            self._last_g = 0.0
            self._last_update_t = t_s
            return SpeakerControl(g=0.0, v_limit_v=float(getattr(u, "spk_v_limit_v", 1.0)), f_mode=int(getattr(u, "spk_mode", 0)))

        if (t_s - self._last_update_t) < self.update_interval_s:
            return SpeakerControl(g=self._last_g, v_limit_v=float(getattr(u, "spk_v_limit_v", 1.0)), f_mode=int(getattr(u, "spk_mode", 0)))

        # We interpret requested volume as a minimum loudness proxy in [0,1].
        loud_req = vol_req

        best_g = self._last_g
        best_j = float("inf")

        n = max(5, int(self.grid_points))
        for i in range(n):
            g = i / (n - 1)

            p_spk, _p_joule, loud = speaker_expected_power_w(
                self.params,
                state,
                SpeakerControl(g=g, v_limit_v=float(getattr(u, "spk_v_limit_v", 1.0)), f_mode=int(getattr(u, "spk_mode", 0))),
                v_bat_v=v_bat_v,
            )

            deficit = max(0.0, loud_req - loud)
            j = (
                self.weights.w_loud_deficit * (deficit**2)
                + self.weights.w_power * p_spk
                + self.weights.w_smooth * ((g - self._last_g) ** 2)
            )

            if j < best_j:
                best_j = j
                best_g = g

        self._last_g = _clamp(best_g, 0.0, 1.0)
        self._last_update_t = t_s
        return SpeakerControl(g=self._last_g, v_limit_v=float(getattr(u, "spk_v_limit_v", 1.0)), f_mode=int(getattr(u, "spk_mode", 0)))
