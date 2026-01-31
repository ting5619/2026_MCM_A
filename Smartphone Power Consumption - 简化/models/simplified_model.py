from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from models.types import UsageInputs, clamp01


@dataclass(frozen=True)
class SimplifiedThermalBatteryParams:
    """简化温度耦合模型参数（对应用户给出的新模型）。

    状态：
    - z: SOC（0~1）
    - T_sys: 系统温度（℃）

    说明：
    - 本模型是“系统级 lumped model”，不会再维护每个组件的独立温度/队列/状态机。
    - 功耗与温度的耦合按用户给定公式实现。
    """

    # Battery electrical bounds
    v_min: float = 3.3
    v_max: float = 4.2

    # Battery nominal capacity
    c_nom_ah: float = 4.5

    # Thermal dynamics
    tau_t_s: float = 600.0  # τ_T
    alpha_heat_c_per_w: float = 0.15  # α_heat
    beta_bat_c_per_w: float = 0.08  # β_bat

    # Temperature-optimum
    t_opt_c: float = 25.0

    # Capacity temperature correction
    k_cold: float = 0.005
    k_hot: float = 0.002

    # Battery voltage temperature correction
    k_voc_per_c: float = 0.003

    # Screen efficiency
    eta_screen_25: float = 0.92
    k_eta_screen_per_c: float = 0.002

    # Network efficiency
    eta_network_25: float = 0.88
    k_eta_network_per_c: float = 0.0015

    # CPU leakage
    cpu_leak_k_per_c: float = 0.03

    # GPS accuracy temperature factor
    gps_sigma_k_per_c: float = 0.01

    # Thermal protection thresholds
    t_warning_c: float = 45.0
    t_throttle_c: float = 55.0
    t_shutdown_c: float = 65.0
    t_heat_c: float = 0.0

    # Heater
    k_heater_w_per_c: float = 0.5

    # --- Power model coefficients (simple affine/linear maps) ---
    # Base power that is always on (PMU, DRAM idle, etc.)
    p_base_w: float = 0.8

    # Screen power ~ k * brightness (0~1)
    k_screen_w_per_l: float = 2.5

    # CPU dynamic power ~ k * utilization (0~1)
    k_cpu_dyn_w_per_u: float = 3.0

    # CPU base leakage reference power at 25C
    p_cpu_base_w: float = 0.6

    # Network lumped power: k_rate * (R / 1e6)
    k_net_w_per_mbps: float = 0.8

    # GPS power: k_gps * (f_update / 1Hz)
    k_gps_w_per_hz: float = 0.25

    # Speaker power: k_spk * volume (0~1)
    k_spk_w_per_v: float = 1.2


@dataclass
class SimplifiedState:
    z: float
    t_sys_c: float


@dataclass(frozen=True)
class SimplifiedOutputs:
    # controls
    ctrl_coeff: float
    l: float
    u: float
    r_mbps: float
    gps_f_hz: float
    v: float

    # temperatures
    t_bat_c: float

    # power breakdown
    p_total_w: float
    p_heat_source_w: float
    p_heater_w: float

    p_screen_w: float
    p_screen_actual_w: float

    p_cpu_dyn_w: float
    p_cpu_leak_w: float
    p_cpu_total_w: float

    p_network_w: float
    p_network_actual_w: float

    p_gps_w: float
    p_speaker_w: float

    # battery terms
    v_oc_v: float
    f_t_bat: float

    # gps metric
    gps_sigma_m: float


def f_t_bat(params: SimplifiedThermalBatteryParams, t_bat_c: float) -> float:
    """电池容量温度修正因子 f_T_bat(T_bat)。"""

    if t_bat_c <= params.t_opt_c:
        f = 1.0 - params.k_cold * (params.t_opt_c - t_bat_c) ** 2
    else:
        f = 1.0 - params.k_hot * (t_bat_c - params.t_opt_c) ** 2

    return max(0.10, float(f))


def v_oc(params: SimplifiedThermalBatteryParams, z: float, t_bat_c: float) -> float:
    """开路电压温度修正：V_oc = [Vmin + (Vmax-Vmin) z] * [1 + 0.003*(T_bat-25)]"""

    z = clamp01(float(z))
    v0 = params.v_min + (params.v_max - params.v_min) * z
    return max(0.1, v0 * (1.0 + params.k_voc_per_c * (t_bat_c - params.t_opt_c)))


def eta_screen(params: SimplifiedThermalBatteryParams, t_sys_c: float) -> float:
    """屏幕效率温度修正：η_screen(T)=0.92*(1-0.002*(T-25))"""

    eta = params.eta_screen_25 * (1.0 - params.k_eta_screen_per_c * (t_sys_c - params.t_opt_c))
    return max(0.20, float(eta))


def eta_network(params: SimplifiedThermalBatteryParams, t_sys_c: float) -> float:
    """网络模块效率温度修正：η_network(T)=0.88*(1-0.0015*(T-25))"""

    eta = params.eta_network_25 * (1.0 - params.k_eta_network_per_c * (t_sys_c - params.t_opt_c))
    return max(0.20, float(eta))


def f_soc(z: float) -> float:
    """SOC 控制系数：f_SOC = 0.2 + 0.8*SOC"""

    return clamp01(0.2 + 0.8 * clamp01(float(z)))


def f_t_sys(t_sys_c: float) -> float:
    """温度控制系数 f_T(T_sys)（按用户给出的分段规则）。"""

    t = float(t_sys_c)
    if 20.0 <= t <= 30.0:
        f = 1.0 - 0.02 * abs(t - 25.0)
    elif t > 30.0:
        f = 0.8 - 0.05 * (t - 30.0)
    else:
        f = 0.8 - 0.03 * (20.0 - t)
    return clamp01(float(f))


def _f_t_brightness(t_sys_c: float) -> float:
    return clamp01(1.0 - 0.015 * max(0.0, float(t_sys_c) - 35.0))


def _f_t_cpu(t_sys_c: float) -> float:
    return clamp01(1.0 - 0.03 * max(0.0, float(t_sys_c) - 40.0))


def _f_t_network(t_sys_c: float) -> float:
    return clamp01(1.0 - 0.02 * max(0.0, float(t_sys_c) - 45.0))


def _f_t_speaker(t_sys_c: float) -> float:
    return clamp01(1.0 - 0.01 * max(0.0, float(t_sys_c) - 40.0))


def _requests_from_usage(u_req: UsageInputs) -> tuple[float, float, float, float, float]:
    l_req = clamp01(float(getattr(u_req, "brightness", 0.0)))
    u_req_cpu = clamp01(float(getattr(u_req, "cpu_demand", 0.0)))

    wifi_bps = max(0.0, float(getattr(u_req, "wifi_rate_bps", 0.0)))
    cell_bps = max(0.0, float(getattr(u_req, "cell_arrival_bps", 0.0)))
    bt_bps = max(0.0, float(getattr(u_req, "bt_arrival_bps", 0.0)))

    r_req_mbps = (wifi_bps + cell_bps + bt_bps) / 1e6

    f_req_hz = max(0.0, float(getattr(u_req, "gps_update_min_hz", 0.0)))
    v_req = clamp01(float(getattr(u_req, "speaker_volume", 0.0)))

    return l_req, u_req_cpu, r_req_mbps, f_req_hz, v_req


def compute_controls(
    params: SimplifiedThermalBatteryParams,
    state: SimplifiedState,
    u_req: UsageInputs,
) -> tuple[float, float, float, float, float, float]:
    l_req, u_req_cpu, r_req_mbps, f_req_hz, v_req = _requests_from_usage(u_req)

    base = min(f_soc(state.z), f_t_sys(state.t_sys_c))

    if state.t_sys_c > params.t_throttle_c:
        base *= 0.5

    base = clamp01(base)

    l = l_req * min(base, _f_t_brightness(state.t_sys_c))
    u = u_req_cpu * min(base, _f_t_cpu(state.t_sys_c))
    r = r_req_mbps * min(base, _f_t_network(state.t_sys_c))

    if state.t_sys_c > 50.0:
        f_update = 0.0
    else:
        f_update = f_req_hz * base

    v = v_req * min(base, _f_t_speaker(state.t_sys_c))

    return base, clamp01(l), clamp01(u), max(0.0, r), max(0.0, f_update), clamp01(v)


def _solve_algebraic(
    params: SimplifiedThermalBatteryParams,
    state: SimplifiedState,
    u_req: UsageInputs,
    use_control: bool,
) -> SimplifiedOutputs:
    if use_control:
        ctrl_coeff, l, u, r_mbps, gps_f_hz, v = compute_controls(params, state, u_req)
    else:
        l_req, u_req_cpu, r_req_mbps, f_req_hz, v_req = _requests_from_usage(u_req)
        ctrl_coeff = 1.0
        l, u, r_mbps, gps_f_hz, v = l_req, u_req_cpu, r_req_mbps, f_req_hz, v_req

    p_screen = params.k_screen_w_per_l * l
    p_cpu_dyn = params.k_cpu_dyn_w_per_u * u
    p_network = params.k_net_w_per_mbps * r_mbps
    p_gps = params.k_gps_w_per_hz * gps_f_hz
    p_speaker = params.k_spk_w_per_v * v

    p_cpu_leak = params.p_cpu_base_w * math.exp(params.cpu_leak_k_per_c * (state.t_sys_c - params.t_opt_c))
    p_cpu_total = p_cpu_dyn + p_cpu_leak

    p_screen_actual = p_screen / eta_screen(params, state.t_sys_c)
    p_network_actual = p_network / eta_network(params, state.t_sys_c)

    p_heater = 0.0
    if state.t_sys_c < params.t_heat_c and state.z > 0.3:
        p_heater = max(0.0, params.k_heater_w_per_c * (params.t_opt_c - state.t_sys_c))

    p_total = params.p_base_w + p_screen_actual + p_cpu_total + p_network_actual + p_gps + p_speaker + p_heater
    t_bat = state.t_sys_c + params.beta_bat_c_per_w * p_total

    for _ in range(3):
        voc_v = v_oc(params, state.z, t_bat)
        fcap = f_t_bat(params, t_bat)
        p_total = params.p_base_w + p_screen_actual + p_cpu_total + p_network_actual + p_gps + p_speaker + p_heater
        t_bat = state.t_sys_c + params.beta_bat_c_per_w * p_total

    voc_v = v_oc(params, state.z, t_bat)
    fcap = f_t_bat(params, t_bat)

    gps_sigma_base = max(0.5, float(getattr(u_req, "gps_sigma_max_m", 10.0)))
    gps_sigma = gps_sigma_base * (1.0 + params.gps_sigma_k_per_c * abs(state.t_sys_c - params.t_opt_c))

    p_heat_source = params.p_base_w + p_screen + p_cpu_dyn + p_cpu_leak + p_network + p_gps + p_speaker

    return SimplifiedOutputs(
        ctrl_coeff=float(ctrl_coeff),
        l=float(l),
        u=float(u),
        r_mbps=float(r_mbps),
        gps_f_hz=float(gps_f_hz),
        v=float(v),
        t_bat_c=float(t_bat),
        p_total_w=float(p_total),
        p_heat_source_w=float(p_heat_source),
        p_heater_w=float(p_heater),
        p_screen_w=float(p_screen),
        p_screen_actual_w=float(p_screen_actual),
        p_cpu_dyn_w=float(p_cpu_dyn),
        p_cpu_leak_w=float(p_cpu_leak),
        p_cpu_total_w=float(p_cpu_total),
        p_network_w=float(p_network),
        p_network_actual_w=float(p_network_actual),
        p_gps_w=float(p_gps),
        p_speaker_w=float(p_speaker),
        v_oc_v=float(voc_v),
        f_t_bat=float(fcap),
        gps_sigma_m=float(gps_sigma),
    )


def step(
    params: SimplifiedThermalBatteryParams,
    state: SimplifiedState,
    u_req: UsageInputs,
    t_amb_c: float,
    dt_s: float,
    use_control: bool,
) -> tuple[SimplifiedState, SimplifiedOutputs]:
    out = _solve_algebraic(params, state, u_req, use_control=use_control)

    denom_wh = max(1e-6, params.c_nom_ah * out.v_oc_v * out.f_t_bat)
    dz_dt = -(out.p_total_w / denom_wh) / 3600.0

    dT_dt = (float(t_amb_c) + params.alpha_heat_c_per_w * out.p_heat_source_w - state.t_sys_c) / max(1e-6, params.tau_t_s)

    z_next = clamp01(state.z + dz_dt * dt_s)
    t_next = state.t_sys_c + dT_dt * dt_s

    if t_next >= params.t_shutdown_c:
        z_next = 0.0

    return SimplifiedState(z=float(z_next), t_sys_c=float(t_next)), out


def simulate(
    params: SimplifiedThermalBatteryParams,
    usage_fn: Callable[[float], UsageInputs],
    z0: float,
    t0_sys_c: float,
    t_amb_c: float,
    dt_s: float = 1.0,
    t_max_s: float = 24 * 3600,
    z_min: float = 0.02,
    use_control: bool = True,
) -> Tuple[float, Dict[str, list]]:
    state = SimplifiedState(z=clamp01(z0), t_sys_c=float(t0_sys_c))

    traces: Dict[str, list] = {
        "t_s": [],
        "soc": [],
        "t_sys_c": [],
        "t_bat_c": [],
        "ctrl_coeff": [],
        "p_total_w": [],
        "p_heat_source_w": [],
        "p_heater_w": [],
        "v_oc_v": [],
        "f_t_bat": [],
        "l": [],
        "u": [],
        "r_mbps": [],
        "gps_f_hz": [],
        "v": [],
        "p_screen_w": [],
        "p_screen_actual_w": [],
        "p_cpu_dyn_w": [],
        "p_cpu_leak_w": [],
        "p_cpu_total_w": [],
        "p_network_w": [],
        "p_network_actual_w": [],
        "p_gps_w": [],
        "p_speaker_w": [],
        "gps_sigma_m": [],
    }

    t_s = 0.0
    while t_s < t_max_s and state.z > z_min:
        u_req = usage_fn(t_s)
        state, out = step(params, state, u_req, t_amb_c=t_amb_c, dt_s=dt_s, use_control=use_control)

        traces["t_s"].append(t_s)
        traces["soc"].append(state.z)
        traces["t_sys_c"].append(state.t_sys_c)
        traces["t_bat_c"].append(out.t_bat_c)
        traces["ctrl_coeff"].append(out.ctrl_coeff)
        traces["p_total_w"].append(out.p_total_w)
        traces["p_heat_source_w"].append(out.p_heat_source_w)
        traces["p_heater_w"].append(out.p_heater_w)
        traces["v_oc_v"].append(out.v_oc_v)
        traces["f_t_bat"].append(out.f_t_bat)

        traces["l"].append(out.l)
        traces["u"].append(out.u)
        traces["r_mbps"].append(out.r_mbps)
        traces["gps_f_hz"].append(out.gps_f_hz)
        traces["v"].append(out.v)

        traces["p_screen_w"].append(out.p_screen_w)
        traces["p_screen_actual_w"].append(out.p_screen_actual_w)
        traces["p_cpu_dyn_w"].append(out.p_cpu_dyn_w)
        traces["p_cpu_leak_w"].append(out.p_cpu_leak_w)
        traces["p_cpu_total_w"].append(out.p_cpu_total_w)
        traces["p_network_w"].append(out.p_network_w)
        traces["p_network_actual_w"].append(out.p_network_actual_w)
        traces["p_gps_w"].append(out.p_gps_w)
        traces["p_speaker_w"].append(out.p_speaker_w)
        traces["gps_sigma_m"].append(out.gps_sigma_m)

        t_s += dt_s

        if state.z <= 0.0:
            break

    return t_s, traces
