from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models.battery_model import ThermalControl, ThermalNetworkState, clamp01


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


@dataclass(frozen=True)
class ThermalCostWeights:
        """热管理控制的阶段代价权重（加热器 + 热保护节流）。

        目标函数（概念形式）：
            J = a1 (T_bat - T_opt)^2 + a2 (T_cpu - T_cpu_max)^2 + a3 P_heat

        另加：电池温度越界软约束惩罚（T_bat 不希望离开 [t_bat_lo_c, t_bat_hi_c]）。

        参数含义/调参方向：
        - a1 ↑：更偏向把电池温度拉到最优区间（低温时更愿意加热；耗电↑）
        - a2 ↑：更偏向限制 CPU 高温（可能更“保守”，节流更早/更强）
        - a3 ↑：更惩罚加热功率（更省电但低温下性能/容量可能受影响）
        """

        a1: float = 1.0  # 电池温度跟踪 T_opt 的权重
        a2: float = 1.0  # CPU 过热惩罚权重
        a3: float = 0.25  # 加热功率惩罚权重（乘在 P_heat[W] 上）

        t_opt_c: float = 25.0  # 电池“最优”温度设定点（℃）
        t_cpu_max_c: float = 85.0  # CPU 温度参考上限（℃），用于惩罚项中心

        t_bat_lo_c: float = 0.0  # 电池温度软下界（℃）
        t_bat_hi_c: float = 45.0  # 电池温度软上界（℃）
        w_bat_bounds: float = 8.0  # 越界惩罚权重（越大越不允许越界）


@dataclass(frozen=True)
class ThermalManagerParams:
    """热管理执行器参数（加热器 + 热节流保护）。

    这些参数不直接定义优化权重，而是定义：
    - 加热器输出上限/效率模型
    - 低 SOC 时加热门限
    - 外壳/CPU 温度触发节流的阈值
    """

    # Heater（加热器）
    k_h: float = 2.5  # W/°C：低温时把 (T_opt - T_bat) 映射成基线加热功率的比例系数
    soc_safe: float = 0.10  # SOC 安全阈值（比例0~1）；SOC 太低时抑制/禁止加热

    eta0: float = 0.92  # 名义电-热效率（电功率→热功率）
    gamma_eta: float = 0.35  # 低温效率衰减强度（越大表示越冷效率越差）

    p_heat_max_w: float = 5.0  # 最大“输送到电池”的热功率上限（W）

    # Hot throttling（过热节流，输出 throttle_factor 影响其他子系统需求/上限）
    t_case_soft_c: float = 45.0  # 外壳“软阈值”（℃）：低于该值不节流
    t_case_hard_c: float = 55.0  # 外壳“硬阈值”（℃）：高于该值强制更强节流

    t_cpu_lvl1_c: float = 75.0  # CPU 节流等级1阈值（℃）
    t_cpu_lvl2_c: float = 85.0  # CPU 节流等级2阈值（℃）
    t_cpu_lvl3_c: float = 95.0  # CPU 节流等级3阈值（℃）


class ThermalOneStepController:
    """One-step controller for heater power and global throttling.

    - Chooses delivered heater heat P_heat (W) using a discrete candidate set.
    - Computes throttle_factor in [0,1] for hot protection.

    This controller is intentionally simple and fast so it can run inside
    simulate_soc and the rollout planner.
    """

    def __init__(
        self,
        params: ThermalManagerParams = ThermalManagerParams(),
        weights: ThermalCostWeights = ThermalCostWeights(),
        update_interval_s: float = 5.0,
        p_heat_candidates_w: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0),
    ) -> None:
        """创建热管理控制器。

        参数：
        - params：执行器/保护阈值参数（加热器效率、功率上限、节流阈值等）
        - weights：优化权重（电池温度/CPU温度/加热功率的权衡）
        - update_interval_s：加热功率的更新周期（秒）；节流因子每步都会计算
        - p_heat_candidates_w：离散加热功率候选集（单位 W，表示“输送到电池的热功率”）
        """
        self.params = params
        self.weights = weights
        self.update_interval_s = update_interval_s
        self.p_heat_candidates_w = p_heat_candidates_w
        self._last_update_t = -1e18
        self._last_p_heat_w = 0.0

    def reset(self, p_heat0_w: float = 0.0) -> None:
        self._last_update_t = -1e18
        self._last_p_heat_w = max(0.0, p_heat0_w)

    def heater_efficiency(self, t_bat_c: float) -> float:
        # eta_heat(T) = eta0 * (1 - gamma * (Topt - Tbat)/Topt)
        t_opt = self.weights.t_opt_c
        eta = self.params.eta0 * (1.0 - self.params.gamma_eta * max(0.0, (t_opt - t_bat_c) / max(1e-6, t_opt)))
        return _clamp(eta, 0.20, 1.0)

    def _stage_cost(self, t_cpu_c: float, t_bat_c: float, p_heat_w: float) -> float:
        j = 0.0
        j += self.weights.a1 * ((t_bat_c - self.weights.t_opt_c) ** 2)
        j += self.weights.a2 * ((t_cpu_c - self.weights.t_cpu_max_c) ** 2)
        j += self.weights.a3 * max(0.0, p_heat_w)

        # Soft bounds for battery temperature range
        lo_excess = max(0.0, self.weights.t_bat_lo_c - t_bat_c)
        hi_excess = max(0.0, t_bat_c - self.weights.t_bat_hi_c)
        j += self.weights.w_bat_bounds * (lo_excess**2 + hi_excess**2)
        return j

    def _throttle_factor(self, t_cpu_c: float, t_case_c: float) -> float:
        # Case-based throttling (linear between soft and hard)
        if t_case_c <= self.params.t_case_soft_c:
            f_case = 1.0
        elif t_case_c >= self.params.t_case_hard_c:
            f_case = 0.5
        else:
            frac = (t_case_c - self.params.t_case_soft_c) / max(1e-6, self.params.t_case_hard_c - self.params.t_case_soft_c)
            f_case = 1.0 - 0.5 * _clamp(frac, 0.0, 1.0)

        # CPU staged protection (as specified)
        f_cpu = 1.0
        if t_cpu_c > self.params.t_cpu_lvl1_c:
            f_cpu = min(f_cpu, 0.70)
        if t_cpu_c > self.params.t_cpu_lvl2_c:
            f_cpu = min(f_cpu, 0.50)
        if t_cpu_c > self.params.t_cpu_lvl3_c:
            f_cpu = min(f_cpu, 0.20)

        return _clamp(min(f_case, f_cpu), 0.0, 1.0)

    def __call__(
        self,
        t_s: float,
        therm: ThermalNetworkState,
        soc: float,
        t_env_c: float,
        p_cpu_w: float,
        p_bat_heat_wo_heater_w: float,
        dt_s: float,
    ) -> ThermalControl:
        """计算当前时刻的热管理输出（加热功率 + 节流因子）（逐参数说明）。

        参数：
        - t_s：当前仿真时间（s）
        - therm：热网络状态（含 t_bat_c / t_cpu_c / t_case_c 等，单位℃）
        - soc：电池 SOC（比例 0~1）；低 SOC 时会抑制加热
        - t_env_c：环境温度（℃），用于上层统一接口（此控制器自身不直接使用）
        - p_cpu_w：当前 CPU 功耗（W），用于一步温度预测/过热风险评估
        - p_bat_heat_wo_heater_w：电池“非加热器来源”的发热功率（W，如 I^2R 等）
        - dt_s：仿真步长（s），用于一步温度预测

        返回：
        - ThermalControl：p_heat_w（输送到电池的热功率，W）、p_heat_elec_w（电功率，W）、throttle_factor（0~1）
        """
        # Always compute throttle (protection)
        throttle = self._throttle_factor(therm.t_cpu_c, therm.t_case_c)

        # Heating disabled when hot enough
        if therm.t_bat_c >= self.weights.t_opt_c:
            self._last_p_heat_w = 0.0
            self._last_update_t = t_s
            return ThermalControl(p_heat_w=0.0, p_heat_elec_w=0.0, throttle_factor=throttle)

        # Heater update timing
        if (t_s - self._last_update_t) < self.update_interval_s:
            eta = self.heater_efficiency(therm.t_bat_c)
            p_elec = self._last_p_heat_w / max(1e-6, eta)
            return ThermalControl(p_heat_w=self._last_p_heat_w, p_heat_elec_w=p_elec, throttle_factor=throttle)

        # Low-temp heating mode (baseline law)
        soc_gate = min(1.0, clamp01(soc) / max(1e-6, self.params.soc_safe))
        p_base = self.params.k_h * max(0.0, (self.weights.t_opt_c - therm.t_bat_c)) * soc_gate
        p_base = _clamp(p_base, 0.0, self.params.p_heat_max_w)

        # One-step selection over candidates around p_base
        # We approximate next-step temps by nudging T_bat toward T_opt with heater.
        best_p = 0.0
        best_j = float("inf")

        for p in self.p_heat_candidates_w:
            p = _clamp(p, 0.0, self.params.p_heat_max_w)
            # Bias toward baseline
            if abs(p - p_base) > 3.0:
                continue

            # Cheap prediction: 
            # dT_bat ~ (P_bat_heat + P_heat) * dt / C_eff, with C_eff tuned by bounds
            c_eff = 25.0
            t_bat_pred = therm.t_bat_c + ((p_bat_heat_wo_heater_w + p) / max(1e-6, c_eff)) * dt_s
            t_cpu_pred = therm.t_cpu_c + (p_cpu_w / max(1e-6, 20.0)) * dt_s

            j = self._stage_cost(t_cpu_pred, t_bat_pred, p)
            if j < best_j:
                best_j = j
                best_p = p

        self._last_p_heat_w = best_p
        self._last_update_t = t_s
        eta = self.heater_efficiency(therm.t_bat_c)
        p_elec = best_p / max(1e-6, eta)
        return ThermalControl(p_heat_w=best_p, p_heat_elec_w=p_elec, throttle_factor=throttle)
