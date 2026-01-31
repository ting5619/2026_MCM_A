from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


@dataclass(frozen=True)
class BatteryParams:
    # Electrical
    v_min: float = 3.3
    v_max: float = 4.2
    eta_pmu: float = 0.9

    # Capacity
    c_nom_ah: float = 4.5  # 4500 mAh
    aging_alpha: float = 0.0  # capacity fade fraction, e.g. 0.2 means -20%

    # Self-discharge / leakage (equiv current)
    i_sd_a: float = 0.0

    # Temperature-capacity correction
    t_ref_c: float = 25.0
    t_cold_c: float = 0.0
    g_cold: float = 0.75  # capacity multiplier at t_cold_c


@dataclass(frozen=True)
class BatterySocParams:
    """Battery SOC/OCV/internal-loss/aging model per the user's equations.

    SOC is tracked as a fraction z in [0,1] in simulate_soc, but the model
    uses SOC_pct = 100*z.
    """

    # Nominal energy corresponding to Q_nom_ah at v_nom_v
    q_nom_ah: float = 2.0
    v_nom_v: float = 4.0

    # OCV model coefficients (v_soc in [0,1] V)
    b11: float = 0.10
    b12: float = 2.2
    b13: float = -0.05
    b14: float = 0.12
    b15: float = -0.18
    b16: float = 0.95
    b17: float = 3.3

    # Internal resistance model coefficients
    b21: float = 0.08
    b22: float = -1.8
    b23: float = 0.035

    # Temperature dependence for total resistance R_total(T)
    # R_total(T) = R_soc(SOC) * exp[(Ea/Rg) * (1/T - 1/T0)]
    # with T in Kelvin.
    ea_over_rg_k: float = 1800.0
    t0_k: float = 298.15

    # Discharge efficiency exponent: eta(i) = 1/(i^k_d)
    k_d: float = 0.08
    i_eps_a: float = 1e-3

    # Battery temperature algebraic model parameters
    r_cpu_env: float = 10.0
    r_bat_env: float = 16.0
    r_cpu_bat: float = 4.0
    tau_t_bat_s: float = 60.0  # smoothing time constant

    # Aging model
    k_aging: float = 2.0e-9
    alpha1: float = 1.0
    alpha2: float = -2500.0
    alpha3: float = 1.0
    alpha4: float = -2500.0
    tau_soc_avg_s: float = 1800.0

    # Usable capacity correction vs battery temperature (piecewise quadratic)
    t_opt_c: float = 25.0
    beta1_per_c2: float = 0.005
    beta2_per_c2: float = 0.002


@dataclass
class BatterySocState:
    t_bat_c: float = 25.0
    q_max_ah: float = 2.0
    soc_avg: float = 1.0  # fraction in [0,1]


@dataclass(frozen=True)
class ThermalParams:
    # Simple 1st-order lumped phone temperature model
    c_th_j_per_c: float = 35.0  # effective heat capacity (J/°C)
    r_th_c_per_w: float = 6.0   # thermal resistance (°C/W)


@dataclass(frozen=True)
class ThermalNetworkParams:
    """3-node thermal network: CPU, Battery, and Case.

    This implements the user's coupled ODEs:
      C_cpu dT_cpu/dt = P_cpu - (T_cpu-T_case)/R_cpu_case - (T_cpu-T_bat)/R_cpu_bat
      C_bat dT_bat/dt = P_bat_heat + P_heat - (T_bat-T_case)/R_bat_case - (T_bat-T_env)/R_bat_env
      C_case dT_case/dt = (T_cpu-T_case)/R_cpu_case + (T_bat-T_case)/R_bat_case - Q_case_env

    Q_case_env includes convection + radiation.
    """

    # Heat capacities
    c_th_cpu_j_per_c: float = 12.0
    c_th_bat_j_per_c: float = 25.0
    c_th_case_j_per_c: float = 45.0

    # Conductive thermal resistances
    r_th_cpu_case_c_per_w: float = 5.0
    r_th_cpu_bat_c_per_w: float = 10.0
    r_th_bat_case_c_per_w: float = 8.0
    r_th_bat_env_c_per_w: float = 18.0

    # Case <-> environment heat exchange
    a_case_m2: float = 0.020
    h_c_w_per_m2_c: float = 7.0
    emissivity: float = 0.85


@dataclass
class ThermalNetworkState:
    t_cpu_c: float = 25.0
    t_bat_c: float = 25.0
    t_case_c: float = 25.0


@dataclass(frozen=True)
class ThermalControl:
    """Thermal management outputs.

    p_heat_w is *delivered heat* into the battery module (W).
    throttle_factor scales down performance-oriented demands in hot states.
    """

    # Delivered heat into battery module
    p_heat_w: float = 0.0
    # Electrical power drawn from battery to realize p_heat_w
    p_heat_elec_w: float = 0.0
    throttle_factor: float = 1.0


def _q_case_to_env_w(net: ThermalNetworkParams, t_case_c: float, t_env_c: float) -> float:
    """Heat flow from case to environment (positive means cooling to env)."""
    import math

    a = max(1e-6, net.a_case_m2)
    h = max(0.0, net.h_c_w_per_m2_c)
    # Convection: q = h*A*(Tcase - Tenv)
    q_conv = h * a * (t_case_c - t_env_c)

    # Radiation: q = eps*sigma*A*(Tcase^4 - Tenv^4)
    sigma_sb = 5.670374419e-8
    eps = _clamp(net.emissivity, 0.0, 1.0)
    t_case_k = max(1.0, t_case_c + 273.15)
    t_env_k = max(1.0, t_env_c + 273.15)
    q_rad = eps * sigma_sb * a * (t_case_k**4 - t_env_k**4)

    return q_conv + q_rad


def thermal_network_step(
    net: ThermalNetworkParams,
    s: ThermalNetworkState,
    t_env_c: float,
    p_cpu_w: float,
    p_bat_heat_w: float,
    p_heat_w: float,
    dt_s: float,
) -> ThermalNetworkState:
    """One explicit Euler step of the coupled thermal network."""

    c_cpu = max(1e-6, net.c_th_cpu_j_per_c)
    c_bat = max(1e-6, net.c_th_bat_j_per_c)
    c_case = max(1e-6, net.c_th_case_j_per_c)

    r_cc = max(1e-6, net.r_th_cpu_case_c_per_w)
    r_cb = max(1e-6, net.r_th_cpu_bat_c_per_w)
    r_bc = max(1e-6, net.r_th_bat_case_c_per_w)
    r_be = max(1e-6, net.r_th_bat_env_c_per_w)

    t_cpu = s.t_cpu_c
    t_bat = s.t_bat_c
    t_case = s.t_case_c

    # CPU node
    dT_cpu_dt = (p_cpu_w - (t_cpu - t_case) / r_cc - (t_cpu - t_bat) / r_cb) / c_cpu

    # Battery node
    dT_bat_dt = (
        (p_bat_heat_w + max(0.0, p_heat_w))
        - (t_bat - t_case) / r_bc
        - (t_bat - t_env_c) / r_be
    ) / c_bat

    # Case node
    q_case_env = _q_case_to_env_w(net, t_case, t_env_c)
    dT_case_dt = (
        (t_cpu - t_case) / r_cc
        + (t_bat - t_case) / r_bc
        - q_case_env
    ) / c_case

    return ThermalNetworkState(
        t_cpu_c=t_cpu + dT_cpu_dt * dt_s,
        t_bat_c=t_bat + dT_bat_dt * dt_s,
        t_case_c=t_case + dT_case_dt * dt_s,
    )


@dataclass(frozen=True)
class ComponentPowers:
    # GPS
    p_gps_track_w: float = 0.12
    e_gps_acq_j: float = 6.0

    # WiFi
    p_wifi_idle_w: float = 0.08
    e_wifi_bit_j: float = 2.5e-8
    e_wifi_scan_j: float = 0.8

    # Bluetooth
    p_bt_w: float = 0.01

    # Speaker
    p_spk_base_w: float = 0.02
    k_spk_vol_w: float = 1.0  # multiplied by volume in [0,1]

    # Background/OS baseline
    p_base_w: float = 0.25


@dataclass(frozen=True)
class ScreenParams:
     """屏幕（OLED）功耗-热-亮度的降阶连续时间模型参数。

     这份代码实现的是“可用于秒级步长仿真”的降阶模型，用来对齐你给的屏幕模型结构：

     1) 控制输入 u(t)（被控量，对应 UsageInputs 中字段）
         - L_cmd(t) / L(t): `UsageInputs.brightness`（归一化亮度命令，0~1）
         - γ(t): `UsageInputs.screen_gamma`（伽马校正指数，>0）
         - f_r(t): `UsageInputs.screen_refresh_hz`（刷新率，Hz）
         - R(t): `UsageInputs.screen_res_scale`（分辨率缩放因子，代码里用 (R/R_max)^2 近似）
         - A_active(t): `UsageInputs.screen_active_area`（激活区域比例，0~1）

     2) 状态变量 x(t)（对应 ScreenState）
         - T_s(t): `ScreenState.t_s_c`（屏幕温度，℃；不同于环境温度）
         - Q_pixel(t): `ScreenState.q_pixel`（像素阵列电荷状态/开关电荷“代理变量”）
         - L_eff(t): `ScreenState.l_eff`（有效亮度，考虑响应时间常数后的实际亮度）

     3) 输出变量 y(t)
         - P_screen(t): `screen_power_w(...)` 的返回值（屏幕瞬时功耗，W）
         - I_screen(t): 代码中通常用电池侧电流 `i_bat_a` 间接体现；若需要可按
            I_screen = P_screen / (η_conv(T_s) * V_bat) 计算。

     4) 功耗产生机理（与你给的分解一致）
         P_screen = P_backlight + P_driver + P_leakage
         - P_backlight：OLED自发光功耗（这里用 L_eff^γ 与温度指数项的简化模型）
         - P_driver：驱动/扫描/刷新功耗（与刷新率/分辨率/激活面积相关）
         - P_leakage：温度相关的漏电功耗

     5) 核心微分方程（代码实现的降阶形式）
         - 亮度响应：dL_eff/dt = (L_cmd - L_eff)/tau_response
         - 温度动态：C_th dT_s/dt = P_screen - (T_s-T_env)/R_th + κ(T_cpu - T_s)
         - 像素电荷代理：dQ_pixel/dt = k_q * f_r * L_eff - Q_pixel/tau_q

     说明：
     - 你给的“dP_backlight/dt 展开式”属于更高阶的“功率随亮度与温度耦合的严格微分形式”。
        本仓库为了可控性与数值稳定性，选用“先算 P，再用热方程推进 T”的常见降阶实现。
     """

     # Geometry / activity
     a_screen_m2: float = 0.012  # ~ 6.2" phone: order 0.01 m^2

     # OLED emission (backlight-equivalent) model
     p_oled_max_w: float = 2.2  # approx max emission power at L=1, gamma~2
     theta_per_c: float = 0.010  # exp(theta*(Ts-25)) temperature coefficient

     # Driver model (refresh + resolution scaling)
     p_driver_60hz_w: float = 0.45  # driver power at 60Hz, high-res, full active area

     # Leakage model
     p_leak_25c_w: float = 0.05
     k_leak_per_c: float = 0.02

     # L_eff response
     tau_response_s: float = 0.25

     # Pixel charge proxy dynamics (not directly used in P_screen by default)
     tau_q_s: float = 3.0
     k_q: float = 1.0

     # Thermal network for screen
     c_th_j_per_c: float = 10.0
     r_th_c_per_w: float = 12.0
     kappa_cpu_w_per_c: float = 0.10  # coupling gain from CPU/phone temperature

     # Power conversion
     eta0_conv: float = 0.92
     lambda_eta_per_c: float = 0.002
     t_opt_c: float = 25.0


@dataclass
class ScreenState:
    t_s_c: float = 25.0
    q_pixel: float = 0.0
    l_eff: float = 0.0


@dataclass(frozen=True)
class CpuParams:
    """CPU（多核DVFS）功耗-温度-负载的降阶连续时间模型参数。

    该模型对应你给的 CPU 状态空间结构，并在代码中做了可仿真的简化：

    1) 控制输入 u(t)（对应 CpuControl）
       - f_c(t): `CpuControl.f_ghz`（核心频率，GHz）
       - V_dd(t): 由 DVFS 关系 `V_dd(f)` 计算（见 v_min/dvfs_a/dvfs_beta）
       - N_active(t): `CpuControl.n_active`（激活核心数，0..n_total）
       - u_cmd(t): `CpuControl.u_cmd`（命令利用率，0~1；表示想让CPU多忙）

    2) 状态变量 x(t)（对应 CpuState）
       - T_j(t): `CpuState.t_j_c`（芯片结温/核心温度，℃）
       - Q_thermal(t): `CpuState.q_thermal_j`（热缓存/热量累积的代理变量，J）
       - u_eff(t): `CpuState.u_eff`（有效利用率：考虑响应时间常数与IPC变化的等效负载）

    3) 输出变量 y(t)
       - P_cpu(t): `cpu_total_power_w(...)`（CPU瞬时功耗，W）
       - I_cpu(t): 近似为 P_cpu / V_dd（若要与电池侧一致，还需乘 PMU 效率/电压换算）
       - T_j(t): `CpuState.t_j_c`

    4) 功耗分解（与你给的结构一致）
       P_total = P_dynamic + P_static + P_clock
       - 动态功耗：C_eff * N_active * f * V_dd^2 * u_eff
       - 静态/漏电：K_leak * N_active * V_dd * exp(kappa*T/T0)
       - 时钟功耗：C_clock * N_active * f * V_dd^2

    5) 温度动态（热模型，降阶）
       C_th * dT_j/dt = P_total - (T_j - T_amb)/R_th - h*(T_j - T_amb) - εσA(T^4 - Tamb^4)
       其中辐射项在本仓库通常做了简化/可选实现（以避免大步长时不稳定）。

    6) 热节流（throttling）
       通过 f_max_throttled(T_j) 实现：当温度超过 t_throttle_c，动态降低可用频率上限。
    """

    # Topology
    n_total: int = 8

    # Frequency limits
    f_min_ghz: float = 0.3
    f_rated_ghz: float = 2.8

    # DVFS Vdd(f) = Vmin + a*(f - fmin)^beta
    v_min: float = 0.45
    dvfs_a: float = 0.028
    dvfs_beta: float = 0.70

    # Dynamic power coefficient (effective switching capacitance)
    c_eff: float = 0.20  # W / (core*GHz*V^2) after lumping

    # Clock distribution power coefficient
    c_clock: float = 0.03  # W / (core*GHz*V^2)

    # Leakage power coefficient
    k_leak: float = 0.03  # W / (core*V) at reference temp
    kappa_leak: float = 2.2  # temperature exponent coefficient
    t0_k: float = 300.0

    # Workload response time
    tau_workload_s: float = 0.05

    # Thermal network
    c_th_j_per_c: float = 12.0
    r_th_c_per_w: float = 6.0
    h_cool_w_per_c: float = 0.0  # additional linear cooling (optional)

    # Thermal throttling
    t_throttle_c: float = 85.0
    gamma_throttle_per_c: float = 0.015

    # Thermal buffer (Q_thermal) proxy
    tau_q_s: float = 2.0

    # Performance scaling (normalized)
    perf_scale: float = 1.0


@dataclass
class CpuState:
    t_j_c: float = 25.0
    q_thermal_j: float = 0.0
    u_eff: float = 0.0


@dataclass(frozen=True)
class CpuControl:
    f_ghz: float
    n_active: int
    u_cmd: float


@dataclass(frozen=True)
class CellularParams:
    """蜂窝（Cellular Modem）“连续-离散混合”功耗模型参数（CTMC + 连续状态）。

     与你给的蜂窝状态机一致：S0~S4 组成连续时间马尔可夫链（CTMC），概率向量 p(t)
     通过 dp/dt = Λ(t) p 演化。

     1) 控制输入 u(t)（代码里用简化的“速率命令”控制）
         - R_data(t): `CellularControl.r_cmd_bps`（期望发送/服务速率，bps）
         你给的 SNR_target/MCS/B(t) 在本仓库是“折算进等效链路容量/能耗系数”的降阶处理。

     2) 状态变量 x(t)（对应 CellularState）
         - p0..p4：离散状态概率（IDLE/CONNECTED/LOW/HIGH/SEARCH）
         - P_tx(t): `CellularState.p_tx_w`（发射功率，W；由闭环控制/时间常数 tau_tx_s 更新）
         - T_modem(t): `CellularState.t_modem_c`（调制解调器温度，℃）
         - Q_data(t): `CellularState.q_data_bits`（待发送队列长度，bit）
         - X_σ(t): `CellularState.x_sigma_db`（阴影衰落 OU 过程的状态，dB）

     3) 输出变量 y(t)
         - P_radio(t): `p_cell_w`（蜂窝模块瞬时功耗，W）
         - I_radio(t): 可按 P_radio/(η_PS*V_supply) 换算；本项目以电池侧总电流为主。

     4) 功耗分解
         P_radio = P_PA + P_BB + P_RF
         - P_PA 通过 P_tx/η_PA(T,P_tx) 建模
         - P_BB 与速率（bit/s）线性相关（mu_bb_j_per_bit）
         - P_RF 与带宽/温度因子相关
    """

    # Rate limits
    r_max_bps: float = 80e6

    # Bandwidth (fixed for now)
    b_hz: float = 20e6
    b_max_hz: float = 100e6

    # Noise density and receiver noise figure lumped into N0
    n0_w_per_hz: float = 4.0e-21  # ~ -174 dBm/Hz
    nf_lin: float = 6.0

    # Power amplifier and RF/baseband power
    p_bb_base_w: float = 0.25
    mu_bb_j_per_bit: float = 1.0e-8
    p_rf0_w: float = 0.45
    zeta_rf: float = 0.6

    # State-dependent baseline powers (when not in payload transfer)
    p_idle_w: float = 0.06
    p_connected_w: float = 0.20
    p_search_w: float = 0.55

    # PA efficiency model
    eta_max: float = 0.38
    kappa_t: float = 0.004
    t_opt_c: float = 35.0
    xi: float = 2.0
    p_tx_max_w: float = 2.0

    # TX power control (target SNR)
    snr_target_db: float = 8.0
    p_tx_min_w: float = 0.02
    tau_tx_s: float = 0.6

    # Path loss and shadowing (OU process on X_sigma)
    pl0_db: float = 105.0
    theta_x: float = 0.15
    sigma_x_db: float = 0.0  # set >0 if you later add stochastic driving

    # CTMC transition base rates (1/s)
    lambda_search: float = 1 / 180.0
    lambda_timeout: float = 1 / 6.0
    lambda_connect: float = 2.0
    lambda_to_low: float = 4.0
    lambda_to_high: float = 6.0

    # Thresholds
    q_wake_bits: float = 200e3
    r_high_bps: float = 6e6

    # Modem thermal
    c_th_j_per_c: float = 8.0
    r_th_c_per_w: float = 10.0
    kappa_cpu_w_per_c: float = 0.08

    # Queue control heuristic
    target_queue_delay_s: float = 1.5


@dataclass
class CellularState:
    # CTMC probabilities
    p0: float = 1.0
    p1: float = 0.0
    p2: float = 0.0
    p3: float = 0.0
    p4: float = 0.0

    # Continuous states
    p_tx_w: float = 0.05
    t_modem_c: float = 25.0
    q_data_bits: float = 0.0
    x_sigma_db: float = 0.0


@dataclass(frozen=True)
class CellularControl:
    r_cmd_bps: float


@dataclass(frozen=True)
class WiFiParams:
    """WiFi 功耗模型参数（CTMC 状态机 + 连续功率/温度/队列）。

     与你给的 WiFi 状态机一致：OFF/DEEP/LIGHT/IDLE/RX/TX/CCA 共 7 个离散状态，
     状态概率 p(t) 使用 dp/dt = Λ(t)p（并可用离开率 Γ 做等价写法）推进。

     1) 控制输入 u(t)（本仓库的主控量）
         - R_data(t): `WiFiControl.r_cmd_bps`（期望服务速率/发包速率，bps）
         你给的 RSSI_target / P_tx / N_streams / BW / T_beacon 在这里被折算为：
         - 目标 RSSI：`rssi_target_dbm`
         - 发射功率 P_tx：通过一阶惯性 `tau_p_tx_s` 跟踪“满足 RSSI + 速率惩罚”的控制律
         - MIMO/带宽：当前以固定参数 `n_streams`、`bw_hz` 近似（便于 rollout/MPC）

     2) 状态变量 x(t)（对应 WiFiState）
         - p0..p6：离散状态概率
         - P_tx(t): `WiFiState.p_tx_w`（发射功率，W）
         - T_wifi(t): `WiFiState.t_wifi_c`（WiFi芯片温度，℃）
         - Q_wifi(t): `WiFiState.q_wifi_bits`（队列长度，bit）
         - X_f(t): `WiFiState.x_f_db`（快衰落 OU 过程状态，dB）
         - τ_state(t): `WiFiState.tau_state_s`（驻留时间累计量，用于刻画状态“待了多久”）

     3) 输出变量 y(t)
         - P_wifi(t): `p_wifi_w`（WiFi瞬时功耗，W）
         - I_wifi(t): 可按 P_wifi/(η_DCDC*V_bat) 换算；本项目以总电池电流为主。
    """

    # PHY / link
    r_max_bps: float = 250e6
    bw_hz: float = 20e6
    bw_max_hz: float = 160e6
    n_streams: int = 1
    n0_w_per_hz: float = 4.0e-21
    nf_lin: float = 6.0

    # Power control and RSSI model
    rssi_target_dbm: float = -60.0
    p_tx_min_w: float = 0.005
    p_tx_max_w: float = 1.2
    alpha_rssi_w_per_db: float = 0.03
    beta_rate_w: float = 0.08
    r_ref_bps: float = 54e6
    gamma_rate: float = 1.4
    tau_p_tx_s: float = 0.25

    # Pathloss + fading (deterministic OU unless sigma_f_db>0 with external noise)
    pl0_db: float = 75.0
    g_tx_db: float = 0.0
    g_rx_db: float = 0.0
    tau_f_s: float = 2.0
    sigma_f_db: float = 0.0

    # State-dependent baseline powers
    p_deep_sleep_w: float = 0.008
    p_light_sleep_w: float = 0.025
    p_idle_w: float = 0.12
    p_cca_w: float = 0.18

    # RF/BB/clock/leakage parameters
    eta0_pa: float = 0.32
    kappa_t: float = 0.006
    xi: float = 2.2
    p_sat_w: float = 1.0

    p_bb_base_w: float = 0.10
    k_bb_j_per_bit: float = 6.0e-9
    p_rf0_w: float = 0.16
    zeta_rf: float = 0.7
    p_clk0_w: float = 0.03
    p_leak_25c_w: float = 0.012
    k_leak_per_c: float = 0.03

    # CTMC transition rates (1/s)
    tau_deep_wake_s: float = 0.35
    tau_light_wake_s: float = 0.10
    lambda_sleep_down: float = 0.25      # IDLE->LIGHT when no traffic
    lambda_deeper_sleep: float = 0.10    # LIGHT->DEEP when no traffic
    lambda_traffic: float = 6.0          # wakeup trigger gain
    lambda_tx_start: float = 10.0
    lambda_rx_start: float = 4.0
    lambda_busy: float = 2.0
    lambda_back_idle: float = 12.0
    lambda_tx_rx_switch: float = 6.0

    # Thresholds / queue
    q_max_bits: float = 20e6
    q_wake_bits: float = 80e3
    target_queue_delay_s: float = 0.8

    # Thermal
    c_th_j_per_c: float = 6.0
    r_th_c_per_w: float = 14.0
    kappa_cpu_w_per_c: float = 0.06
    kappa_cell_w_per_c: float = 0.03


@dataclass
class WiFiState:
    # CTMC probabilities
    p0: float = 0.0
    p1: float = 1.0
    p2: float = 0.0
    p3: float = 0.0
    p4: float = 0.0
    p5: float = 0.0
    p6: float = 0.0

    # Continuous states
    p_tx_w: float = 0.02
    t_wifi_c: float = 25.0
    q_wifi_bits: float = 0.0
    x_f_db: float = 0.0
    tau_state_s: float = 0.0


@dataclass(frozen=True)
class WiFiControl:
    r_cmd_bps: float


@dataclass(frozen=True)
class GpsParams:
    """GPS 接收机降阶模型参数（模式状态机 + 连续信号质量/精度/温度）。

     对齐你给的 GPS“工作模式状态机”建模：OFF/待机/捕获/跟踪/定位/辅助等模式用概率 m_i 表示。

     1) 控制输入 u(t)（对应 GpsControl；在 simulate_soc 中由 gps_controller 产生）
         - σ_pos_req(t): `GpsControl.sigma_pos_req_m`（精度要求，m，越小越严格）
         - f_update(t): `GpsControl.f_update_hz`（位置更新率，Hz；0 表示关闭）
         - C/N0_thresh、B_loop、assist：用于影响捕获/跟踪/功耗的细节参数

     2) 状态变量 x(t)（对应 GpsState）
         - m0..m5：模式概率（OFF/STANDBY/ACQ/TRACK/NAV/ASSIST）
         - T_GPS(t): `GpsState.t_gps_c`（GPS芯片温度，℃）
         - P_acq(t): `GpsState.p_acq`（捕获进度，0~1）
         - C/N0：`GpsState.cn0_dbhz`（载噪比，dB-Hz）
         - σ_pos_est：`GpsState.sigma_pos_est_m`（当前精度估计，m）
         - N_vis / N_locked / LQ：可见/锁定卫星与锁定质量

     3) 输出变量 y(t)
         - P_GPS(t): `p_gps_w`（GPS瞬时功耗，W）
         - 模式/精度等：`gps_m_*`、`gps_sigma_est_m` 等 traces 字段

     4) 功耗分解（与你给的结构一致）
         P_GPS = P_RF + P_corr + P_DSP + P_memory + P_clock + P_leakage
    """

    # Mode timing / rates
    tau_wake_s: float = 1.2
    tau_sleep_s: float = 4.0
    tau_nav_s: float = 0.25  # average duration of a PVT solve burst

    # Acquisition / lock
    lambda_lock_base: float = 0.35
    lambda_loss_base: float = 0.06
    lq0: float = 0.35
    k_cn0_lock: float = 0.22
    cn0_min_dbhz: float = 20.0
    cn0_ref_dbhz: float = 35.0
    cn0_span_db: float = 10.0

    # Environmental dynamics
    alpha_cn0: float = 0.45
    tau_n_vis_s: float = 25.0

    # LQ dynamics
    tau_lq_s: float = 2.0
    k_lq_cn0: float = 0.28
    k_lq_sat: float = 0.55

    # Accuracy dynamics
    sigma0_m: float = 18.0
    tau_sigma_nav_s: float = 0.8
    tau_sigma_track_s: float = 4.0
    k_sigma_update_m: float = 3.0
    k_sigma_tcxo_m: float = 0.12
    k_proc_gain: float = 0.30  # how much tighter sigma_req improves estimation

    # Correlators / clock
    n_corr_total: int = 2048
    n_corr_track_per_sat: int = 3
    p_corr_per_w: float = 1.2e-5
    f_clk_rel: float = 1.0
    vdd_v: float = 1.05

    # RF power
    p_lna_w: float = 0.015
    p_mixer_w: float = 0.010
    p_adc_w: float = 0.018
    p_if0_w: float = 0.025
    zeta_b_if: float = 0.45
    b_if_max_hz: float = 4.0e6
    f_dop_max_hz: float = 5.0e3

    # DSP / memory / clock
    p_dsp_base_w: float = 0.035
    k_dsp_w: float = 0.006
    beta_sat: float = 0.10
    n_sat_optimal: int = 7

    p_clk0_w: float = 0.006
    f_clk_ref: float = 1.0
    gamma_tcx0: float = 0.0025
    t_ref_c: float = 25.0

    p_mem0_w: float = 0.003
    lambda_mem_access: float = 1.0
    e_access_j: float = 6.0e-6

    # Temperature sensitivity of RF and leakage
    nf0: float = 2.0
    delta_nf_per_c: float = 0.02
    t_nf_opt_c: float = 30.0
    p_leak_25c_w: float = 0.002
    k_leak_per_c: float = 0.04

    # Thermal network
    c_th_j_per_c: float = 6.0
    r_th_c_per_w: float = 16.0
    kappa_shared_w_per_c: float = 0.06

    # PMU efficiency (for current if later needed)
    eta_ldo: float = 0.55
    p_cross_w: float = 0.0015
    eta_dcdc_max: float = 0.90
    p_loss_w: float = 0.0010


@dataclass
class GpsState:
    # CTMC probabilities for modes: OFF, STANDBY, ACQ, TRACK, NAV, ASSIST
    m0: float = 1.0
    m1: float = 0.0
    m2: float = 0.0
    m3: float = 0.0
    m4: float = 0.0
    m5: float = 0.0

    # Continuous states
    t_gps_c: float = 25.0
    p_acq: float = 0.0
    cn0_dbhz: float = 33.0
    sigma_pos_est_m: float = 30.0
    n_vis: float = 8.0
    n_locked: float = 0.0
    lq: float = 0.0


@dataclass(frozen=True)
class GpsControl:
    # Controller-chosen requirements / operating point
    sigma_pos_req_m: float  # [1, 100] meters
    f_update_hz: float      # [0.1, 10] Hz (0 means off)

    # Optional additional knobs (kept for completeness; may be left to defaults)
    cn0_thresh_dbhz: float = 28.0
    b_loop_hz: float = 8.0
    assist: float = 0.0


@dataclass(frozen=True)
class SpeakerParams:
    """扬声器（功放+音圈）电-热耦合降阶模型参数。

     对齐你给的扬声器等效电路与微分方程，但为了适配“秒级步长”的系统级仿真，本仓库采用
     “包络/平均化”近似：不追踪音频载波，只追踪等效 RMS 电平带来的电流、功耗与温升。

     1) 控制输入 u(t)（对应 SpeakerControl / UsageInputs）
         - V_in(t): 用 `UsageInputs.spk_audio_level` 表示归一化音频 RMS（0~1）
         - G(t): `SpeakerControl.g`（数字增益，0~1；由 speaker_controller 输出）
         - V_limit: `SpeakerControl.v_limit_v`（限幅阈值，V）
         - F_mode: `SpeakerControl.f_mode`（滤波/模式：0语音，1音乐）

     2) 状态变量 x(t)（对应 SpeakerState）
         - I_vc(t): `i_vc_a`（音圈电流，A）
         - V_filter(t): `v_filter_v`（LC滤波后等效电压，V）
         - T_vc(t): `t_vc_c`（音圈温度，℃）
         - ∫P_joule dt: `e_joule_j`（焦耳热累积能量，J）

     3) 输出变量 y(t)
         - P_spk(t): traces 的 `p_spk_w`（扬声器/功放总功耗，W）
         - T_vc(t): traces 的 `spk_t_vc_c`（音圈峰值温度，℃）
         - loud_est: `spk_loud_est`（响度代理变量，0~1；便于控制）
    """

    # Electrical (voice coil)
    r_vc0_ohm: float = 4.0
    t0_c: float = 20.0
    alpha_r_per_c: float = 0.00393
    beta_r_per_c2: float = 0.0

    l_vc0_h: float = 420e-6
    i_sat_a: float = 0.9

    k_emf_v_per_a_gamma: float = 0.35
    gamma_emf: float = 0.60

    # Output filter (envelope / averaged)
    l_filter_h: float = 1.2e-3
    c_filter_f: float = 6.0e-3
    r_filter_ohm: float = 0.25

    # Drive / PWM
    v_ref_v: float = 1.0
    eta_buck: float = 0.92
    v_bus_min_v: float = 2.8

    # Class-D efficiency curve
    p_cross_w: float = 0.03
    eta_low: float = 0.50
    eta_max: float = 0.92
    p_tau_w: float = 0.20

    # Switching + conduction loss proxy
    p_sw0_w: float = 0.010
    k_cond_w_per_a2: float = 0.08

    # Idle / standby
    p_idle_w: float = 0.0008
    tau_idle_decay_s: float = 0.35

    # Thermal
    c_th_j_per_c: float = 8.0
    r_th_c_per_w: float = 20.0
    kappa_phone_w_per_c: float = 0.08

    # Loudness proxy
    p_loud_ref_w: float = 0.30
    loudness_gamma: float = 0.55


@dataclass
class SpeakerState:
    i_vc_a: float = 0.0
    v_filter_v: float = 0.0
    t_vc_c: float = 25.0
    e_joule_j: float = 0.0


@dataclass(frozen=True)
class SpeakerControl:
    g: float  # digital gain in [0,1]
    v_limit_v: float = 1.0
    f_mode: int = 0  # 0=voice, 1=music


@dataclass(frozen=True)
class BluetoothParams:
    """蓝牙 BLE 低功耗连接事件模型参数（事件触发脉冲的平均化）。

     对齐你给的 BLE “连接事件（Connection Event）周期性调度”建模：
     - 连接间隔 T_conn（`t_conn_s`）与事件长度 τ_event 共同决定占空比 D_conn = τ_event/T_conn。
     - 本仓库不逐个包仿真，而是用“平均化占空比 + 队列”方式，适配秒级系统仿真。

     1) 控制输入 u(t)
         - 控制器输出平均数据速率命令 `BluetoothControl.r_cmd_bps`（bps）
         - 发射功率控制：如果 `adaptive_power=True`，用一阶惯性 `tau_p_tx_s` 调整 p_tx_dbm

     2) 状态变量 x(t)（对应 BluetoothState）
         - T_BLE(t): `t_ble_c`（蓝牙芯片温度，℃）
         - E_accum(t): `e_accum_j`（能量累积，J）
         - PER(t): `per`（误包率）与 RSSI 平滑估计 `rssi_avg_dbm`
         - Q_bits(t): `q_bits`（待发队列，bit）

     3) 输出变量 y(t)
         - P_BLE(t): traces 的 `p_bt_w`（蓝牙瞬时功耗，W）
         - 连接事件的“活跃程度”在这里通过占空比/速率间接体现。
     """

    # PHY and rate limits
    r_max_bps: float = 2.0e6
    r_phy_bps: float = 1.0e6  # 1M PHY by default (can be extended to 2M/Coded)

    # Connection scheduling
    t_conn_s: float = 0.050
    tau_event_max_s: float = 0.010
    tau_prep_s: float = 0.0012
    tau_turn_s: float = 0.0008

    # Payload constraints
    l_payload_max_bits: float = 251.0 * 8.0
    protocol_overhead_bits: float = 80.0 * 8.0

    # PER/RSSI dynamics
    per_min: float = 1e-4
    per_max: float = 0.35
    tau_per_s: float = 4.0
    tau_rssi_s: float = 1.0
    per_target: float = 0.02
    # Baseline PER vs RSSI: PER0 = sigmoid(a*(rssi_ref - rssi))
    per_sigmoid_a: float = 0.35
    rssi_ref_dbm: float = -70.0

    # RSSI model (short range): RSSI_inst = P_tx_peer - PL(d) + X_s
    p_tx_peer_dbm: float = 0.0
    pl0_db: float = 55.0
    pathloss_gamma: float = 2.6
    d0_m: float = 1.0
    d_m: float = 1.5
    # small-scale fading (OU, deterministic unless sigma_s_db>0)
    omega_s: float = 2.0
    sigma_s_db: float = 0.0

    # TX power control (we keep discrete levels but use a smoothed ODE)
    p_tx_levels_dbm: Tuple[float, ...] = (-20.0, -16.0, -12.0, -8.0, -4.0, 0.0, 4.0, 8.0)
    p_tx_min_dbm: float = -20.0
    p_tx_max_dbm: float = 8.0
    tau_p_tx_s: float = 0.6
    alpha_per_dbm: float = 8.0   # dBm per unit PER error
    beta_rssi_dbm: float = 0.4   # dBm per dB RSSI change
    p_tx_fixed_dbm: float = 0.0
    adaptive_power: bool = True

    # Power model
    p_sleep_25c_w: float = 0.00035
    k_sleep_per_c: float = 0.01
    p_leak_25c_w: float = 0.00020
    k_leak_per_c: float = 0.02

    p_prepare_w: float = 0.010
    p_idle_event_w: float = 0.004
    p_rx_base_w: float = 0.012
    k_rx: float = 0.003
    zeta_bw: float = 0.25
    b_eq_over_ref: float = 1.0

    # PA efficiency
    eta0: float = 0.28
    p_ref_w: float = 0.010

    # Thermal
    c_th_j_per_c: float = 5.0
    r_th_c_per_w: float = 18.0
    kappa_adj_w_per_c: float = 0.05

    # Queue control
    q_max_bits: float = 3e6
    target_queue_delay_s: float = 1.2


@dataclass
class BluetoothState:
    t_ble_c: float = 25.0
    e_accum_j: float = 0.0
    per: float = 0.02
    rssi_avg_dbm: float = -60.0
    x_s_db: float = 0.0
    p_tx_dbm: float = 0.0
    q_bits: float = 0.0
    n_event: int = 0


@dataclass(frozen=True)
class BluetoothControl:
    r_cmd_bps: float


@dataclass(frozen=True)
class UsageInputs:
    # Normalized controls / rates
    brightness: float            # L_cmd in [0,1]
    # CPU workload demand (normalized required compute)
    # Interpretation: demand=1 means "needs" about n_total*f_rated worth of active compute.
    cpu_demand: float            # in [0,1]

    # Screen controls
    screen_gamma: float           # gamma(t) > 0
    screen_refresh_hz: float      # f_r in [60,120]
    screen_res_scale: float       # (R/R_max)^2 in [0,1], high=1, low<1
    screen_active_area: float     # A_active in [0,1]

    gps_fix_duty: float          # D_fix in [0,1]
    gps_acq_rate_hz: float       # lambda_acq in 1/s

    # WiFi offered load (arrival to WiFi queue). Kept as wifi_rate_bps for
    # backward compatibility with scenarios.
    wifi_rate_bps: float         # in [0, +inf)
    wifi_scan_rate_hz: float     # scans per second

    # Cellular offered load (arrival to modem queue)
    cell_arrival_bps: float      # in [0, +inf)

    # Bluetooth offered load (arrival to BLE queue)
    bt_arrival_bps: float        # in [0, +inf)

    bt_on: float                 # in [0,1] (simple duty)

    speaker_volume: float        # in [0,1]

    # GPS (new physics model): app requirements + environment.
    # Placed last (with defaults) to keep dataclass init ordering valid.
    gps_update_min_hz: float = 0.0          # minimum acceptable update rate
    gps_sigma_max_m: float = 50.0           # maximum acceptable position error
    gps_on: float = 0.0                     # app switch: 0=off, 1=on
    gps_cn0_env_dbhz: float = 35.0          # environment-driven equilibrium C/N0
    gps_n_vis_raw: float = 8.0              # visible satellites proxy
    gps_cn0_thresh_dbhz: float = 28.0       # detection / tracking threshold
    gps_b_loop_hz: float = 8.0              # tracking loop bandwidth
    gps_assist: float = 0.0                 # A-GPS assist degree in [0,1]

    # Speaker (new physics model): audio envelope + limiter + mode.
    # speaker_volume remains the *requested loudness* in [0,1].
    spk_audio_level: float = 0.0            # normalized RMS of V_in/V_ref in [0,1]
    spk_v_limit_v: float = 1.0              # limiter threshold in volts
    spk_mode: int = 0                       # 0=voice, 1=music


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def voc(battery: BatteryParams, soc: float) -> float:
    z = clamp01(soc)
    return battery.v_min + (battery.v_max - battery.v_min) * z


def battery_ocv_v(bat: BatterySocParams, soc_frac: float) -> float:
    import math

    soc_pct = 100.0 * clamp01(soc_frac)
    v_soc = (soc_pct / 100.0) * 1.0
    return (
        bat.b11 * math.exp(bat.b12 * v_soc)
        + bat.b13 * (v_soc**4)
        + bat.b14 * (v_soc**3)
        + bat.b15 * (v_soc**2)
        + bat.b16 * v_soc
        + bat.b17
    )


def battery_r_total_ohm(bat: BatterySocParams, soc_frac: float) -> float:
    import math

    soc_pct = 100.0 * clamp01(soc_frac)
    v_soc = (soc_pct / 100.0) * 1.0
    return max(1e-4, bat.b21 * math.exp(bat.b22 * v_soc) + bat.b23)


def battery_r_total_temp_ohm(bat: BatterySocParams, soc_frac: float, t_bat_c: float) -> float:
    """Total resistance with SOC and temperature dependence."""
    import math

    r_soc = battery_r_total_ohm(bat, soc_frac)
    t_k = max(200.0, t_bat_c + 273.15)
    t0_k = max(200.0, bat.t0_k)
    factor = math.exp(_clamp(bat.ea_over_rg_k, 0.0, 20000.0) * (1.0 / t_k - 1.0 / t0_k))
    return max(1e-4, r_soc * factor)


def battery_q_usable_ah(bat: BatterySocParams, q_max_ah: float, t_bat_c: float) -> float:
    """Usable capacity at battery temperature using the provided quadratic correction."""
    t_opt = bat.t_opt_c
    if t_bat_c <= t_opt:
        f = 1.0 - bat.beta1_per_c2 * ((t_opt - t_bat_c) ** 2)
    else:
        f = 1.0 - bat.beta2_per_c2 * ((t_bat_c - t_opt) ** 2)
    return max(1e-9, q_max_ah * _clamp(f, 0.05, 1.0))


def battery_eta_discharge(bat: BatterySocParams, i_b_a: float) -> float:
    # eta(i) = 1/(i^k_d), but clamp to (0,1]
    i = max(bat.i_eps_a, abs(i_b_a))
    eta = 1.0 / (i ** max(0.0, bat.k_d))
    return max(1e-3, min(1.0, eta))


def battery_internal_loss_w(
    bat: BatterySocParams,
    soc_frac: float,
    p_device_w: float,
    t_bat_c: float = 25.0,
) -> Tuple[float, float, float, float]:
    """Return (p_loss_w, i_b_a, v_oc_v, r_total_ohm)."""

    v_oc = max(1e-3, battery_ocv_v(bat, soc_frac))
    r_tot = battery_r_total_temp_ohm(bat, soc_frac, t_bat_c)

    # Solve i_b approximately from power balance with internal losses.
    # Fixed-point iteration: i <- (P_device + P_loss(i))/Voc
    i_b = max(0.0, p_device_w) / v_oc
    p_loss = 0.0
    for _ in range(3):
        eta = battery_eta_discharge(bat, i_b)
        # P_loss = i^2*R + i*v_oc*(1/eta - 1)
        p_ohm = (i_b**2) * r_tot
        p_eff = i_b * v_oc * (1.0 / max(1e-6, eta) - 1.0)
        p_loss = max(0.0, p_ohm + p_eff)
        i_b = (max(0.0, p_device_w) + p_loss) / v_oc

    return p_loss, i_b, v_oc, r_tot


def battery_t_bat_eq_c(bat: BatterySocParams, t_env_c: float, p_cpu_w: float, p_bat_w: float) -> float:
    denom = (bat.r_bat_env + bat.r_cpu_bat + bat.r_cpu_env)
    if denom <= 1e-9:
        return t_env_c
    a = bat.r_cpu_env * bat.r_bat_env / denom
    b = (bat.r_cpu_env * bat.r_bat_env + bat.r_cpu_bat * bat.r_bat_env) / denom
    return t_env_c + a * p_cpu_w + b * p_bat_w


def battery_energy_j(bat: BatterySocParams, q_max_ah: float) -> float:
    # Scale the nominal energy by Q_max/Q_nom.
    e_nom = bat.q_nom_ah * 3600.0 * bat.v_nom_v
    return max(1e-6, e_nom * (max(1e-9, q_max_ah) / max(1e-9, bat.q_nom_ah)))


def battery_aging_step(bat: BatterySocParams, state: BatterySocState, soc_frac: float, t_bat_c: float, dt_s: float) -> BatterySocState:
    import math

    # SOC_avg low-pass
    tau = max(1e-3, bat.tau_soc_avg_s)
    soc_avg_next = state.soc_avg + ((clamp01(soc_frac) - state.soc_avg) / tau) * dt_s
    soc_avg_next = clamp01(soc_avg_next)

    # Aging rate
    t_k = max(200.0, t_bat_c + 273.15)
    rate = bat.k_aging * (
        bat.alpha1 * math.exp(bat.alpha2 / t_k) * soc_avg_next
        + bat.alpha3 * math.exp(bat.alpha4 / t_k) * (1.0 - soc_avg_next)
    )
    dq_dt = -rate * max(1e-9, state.q_max_ah)
    q_next = max(1e-9, state.q_max_ah + dq_dt * dt_s)
    return BatterySocState(t_bat_c=state.t_bat_c, q_max_ah=q_next, soc_avg=soc_avg_next)


def g_temp_capacity(battery: BatteryParams, temp_c: float) -> float:
    # Piecewise linear: at/beyond reference -> 1; at/below cold -> g_cold
    if temp_c >= battery.t_ref_c:
        return 1.0
    if temp_c <= battery.t_cold_c:
        return battery.g_cold
    frac = (temp_c - battery.t_cold_c) / (battery.t_ref_c - battery.t_cold_c)
    return battery.g_cold + (1.0 - battery.g_cold) * frac


def c_eff_ah(battery: BatteryParams, temp_c: float) -> float:
    return battery.c_nom_ah * (1.0 - battery.aging_alpha) * g_temp_capacity(battery, temp_c)


def component_power_w(p: ComponentPowers, u: UsageInputs, temp_c: float) -> Dict[str, float]:
    brightness = clamp01(u.brightness)
    bt_on = clamp01(u.bt_on)
    speaker_volume = clamp01(u.speaker_volume)

    # Legacy GPS lumped model (kept for backward compatibility). If the new
    # GPS physics model is enabled in simulate_soc, this must be excluded from
    # total power to avoid double counting.
    p_gps = u.gps_fix_duty * p.p_gps_track_w + u.gps_acq_rate_hz * p.e_gps_acq_j

    p_wifi = p.p_wifi_idle_w + p.e_wifi_bit_j * u.wifi_rate_bps + u.wifi_scan_rate_hz * p.e_wifi_scan_j

    p_bt = bt_on * p.p_bt_w

    # Legacy speaker power (kept for backward compatibility). If the new
    # speaker physics model is enabled in simulate_soc, this must be excluded
    # from total power to avoid double counting.
    p_spk = p.p_spk_base_w + p.k_spk_vol_w * speaker_volume

    powers = {
        "base": p.p_base_w,
        "gps": p_gps,
        "wifi": p_wifi,
        "bt": p_bt,
        "speaker": p_spk,
    }
    return powers


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _log2(x: float) -> float:
    import math

    return math.log(max(1e-12, x), 2)


def speaker_r_vc_ohm(spk: SpeakerParams, t_vc_c: float) -> float:
    dt = (t_vc_c - spk.t0_c)
    return spk.r_vc0_ohm * (1.0 + spk.alpha_r_per_c * dt + spk.beta_r_per_c2 * (dt**2))


def speaker_l_vc_h(spk: SpeakerParams, i_vc_a: float) -> float:
    x = (i_vc_a / max(1e-6, spk.i_sat_a))
    return spk.l_vc0_h / (1.0 + x * x)


def speaker_dL_dI(spk: SpeakerParams, i_vc_a: float) -> float:
    x = (i_vc_a / max(1e-6, spk.i_sat_a))
    denom = (1.0 + x * x)
    return (-2.0 * spk.l_vc0_h * i_vc_a / max(1e-12, spk.i_sat_a**2)) / (denom * denom)


def speaker_v_emf_v(spk: SpeakerParams, i_vc_a: float) -> float:
    import math

    s = 1.0 if i_vc_a >= 0.0 else -1.0
    return spk.k_emf_v_per_a_gamma * s * (abs(i_vc_a) ** spk.gamma_emf)


def speaker_eta_amp(spk: SpeakerParams, p_out_w: float) -> float:
    import math

    p = max(0.0, p_out_w)
    if p < spk.p_cross_w:
        eta = spk.eta_low * (p / max(1e-9, spk.p_cross_w))
    else:
        eta = spk.eta_max * (1.0 - math.exp(-(p - spk.p_cross_w) / max(1e-9, spk.p_tau_w)))
    return max(0.10, min(0.98, eta))


def speaker_expected_power_w(spk: SpeakerParams, s: SpeakerState, c: SpeakerControl, v_bat_v: float) -> Tuple[float, float, float]:
    """Return (p_total_w, p_joule_w, loudness_est).

    p_total is the battery-side equivalent power of the amplifier+load.
    """

    import math

    r_vc = max(0.2, speaker_r_vc_ohm(spk, s.t_vc_c))
    p_joule = (s.i_vc_a**2) * r_vc

    # Loudness proxy based on Joule heating (dominant acoustic scaling proxy).
    loud = 0.0
    if spk.p_loud_ref_w > 1e-9:
        loud = (max(0.0, p_joule) / spk.p_loud_ref_w) ** spk.loudness_gamma
    loud = max(0.0, min(1.5, loud))

    # Output power proxy (use |V*I| envelope)
    p_out = abs(s.v_filter_v * s.i_vc_a)
    eta = speaker_eta_amp(spk, p_out)

    # Switch + conduction proxy losses
    p_loss = spk.p_sw0_w + spk.k_cond_w_per_a2 * (s.i_vc_a**2)

    p_in = p_out / max(1e-9, eta) + p_loss

    # Supply/buck scaling (if you later want current): keep as power here
    _ = max(spk.v_bus_min_v, spk.eta_buck * max(0.0, v_bat_v))
    return max(0.0, p_in), max(0.0, p_joule), loud


def speaker_step(
    spk: SpeakerParams,
    s: SpeakerState,
    c: SpeakerControl,
    u: UsageInputs,
    t_amb_c: float,
    t_phone_c: float,
    v_bat_v: float,
    dt_s: float,
) -> Tuple[SpeakerState, float, float, float]:
    """Advance speaker states.

    Returns (next_state, p_spk_w, p_joule_w, loudness_est).
    """

    g = clamp01(c.g)
    v_limit = max(0.05, c.v_limit_v)

    # Audio envelope input (RMS-like). If no audio or zero gain, enter idle.
    audio_level = clamp01(getattr(u, "spk_audio_level", 0.0))
    if audio_level <= 1e-4 or g <= 1e-4:
        tau = max(1e-3, spk.tau_idle_decay_s)
        decay = 0.0 if dt_s <= 0.0 else (2.718281828 ** (-dt_s / tau))
        i_next = s.i_vc_a * decay
        v_next = s.v_filter_v * decay

        # Small standby power; Joule heat is from residual current.
        r_vc = max(0.2, speaker_r_vc_ohm(spk, s.t_vc_c))
        p_joule = (i_next**2) * r_vc
        p_spk = spk.p_idle_w + p_joule
        loud = 0.0

        e_next = s.e_joule_j + p_joule * dt_s
        dT_dt = (p_joule - (s.t_vc_c - t_amb_c) / spk.r_th_c_per_w + spk.kappa_phone_w_per_c * (t_phone_c - s.t_vc_c)) / spk.c_th_j_per_c
        t_next = s.t_vc_c + dT_dt * dt_s

        return SpeakerState(i_vc_a=i_next, v_filter_v=v_next, t_vc_c=t_next, e_joule_j=e_next), p_spk, p_joule, loud

    v_in = audio_level * spk.v_ref_v
    v_cmd = max(-v_limit, min(v_limit, g * v_in))

    v_bus = max(spk.v_bus_min_v, spk.eta_buck * max(0.0, v_bat_v))
    duty = clamp01(v_cmd / max(1e-6, spk.v_ref_v))
    v_amp = v_bus * duty

    # Nonlinear elements
    i = s.i_vc_a
    l_vc = speaker_l_vc_h(spk, i)
    dldI = speaker_dL_dI(spk, i)
    v_emf = speaker_v_emf_v(spk, i)

    # Effective inductance and resistance
    l_eq = spk.l_filter_h + l_vc + i * dldI
    l_eq = max(1e-5, l_eq)

    r_vc = max(0.2, speaker_r_vc_ohm(spk, s.t_vc_c))
    i_eps = 1e-3
    r_emf = spk.gamma_emf * spk.k_emf_v_per_a_gamma * (max(i_eps, abs(i)) ** (spk.gamma_emf - 1.0))
    r_eq = spk.r_filter_ohm + r_vc + r_emf

    # Current dynamics
    di_dt = (v_amp - v_emf - r_eq * i) / l_eq
    i_next = i + di_dt * dt_s
    # Keep envelope stable
    i_next = max(-6.0, min(6.0, i_next))

    # Filter capacitor dynamics
    c_f = max(1e-6, spk.c_filter_f)
    dv_dt = (i - (s.v_filter_v - v_emf) / max(0.2, r_vc)) / c_f
    v_next = s.v_filter_v + dv_dt * dt_s
    v_next = max(-v_bus, min(v_bus, v_next))

    # Power + thermal
    s_mid = SpeakerState(i_vc_a=i_next, v_filter_v=v_next, t_vc_c=s.t_vc_c, e_joule_j=s.e_joule_j)
    p_spk, p_joule, loud = speaker_expected_power_w(spk, s_mid, c, v_bat_v=v_bat_v)

    # Energy integral (Joule heat)
    e_next = s.e_joule_j + p_joule * dt_s

    dT_dt = (p_joule - (s.t_vc_c - t_amb_c) / spk.r_th_c_per_w + spk.kappa_phone_w_per_c * (t_phone_c - s.t_vc_c)) / spk.c_th_j_per_c
    t_next = s.t_vc_c + dT_dt * dt_s

    return SpeakerState(i_vc_a=i_next, v_filter_v=v_next, t_vc_c=t_next, e_joule_j=e_next), p_spk, p_joule, loud


def gps_pmu_eff(gps: GpsParams, p_w: float) -> float:
    # η(P) per user's piecewise model (simplified, deterministic)
    if p_w < gps.p_cross_w:
        return gps.eta_ldo
    return gps.eta_dcdc_max * p_w / max(1e-12, (p_w + gps.p_loss_w))


def gps_expected_power_w(gps: GpsParams, s: GpsState, c: GpsControl) -> float:
    import math

    # Clamp controls
    f_update = _clamp(c.f_update_hz, 0.0, 10.0)
    sigma_req = _clamp(c.sigma_pos_req_m, 1.0, 100.0)
    b_loop = _clamp(c.b_loop_hz, 1.0, 20.0)
    assist = _clamp(c.assist, 0.0, 1.0)

    # Mode probabilities
    m0, m1, m2, m3, m4, m5 = s.m0, s.m1, s.m2, s.m3, s.m4, s.m5

    # Correlator activity depends on mode
    n_sat_locked = max(0.0, min(s.n_vis, s.n_locked))
    n_corr_acq = float(gps.n_corr_total)
    n_corr_track = float(gps.n_corr_track_per_sat) * n_sat_locked
    n_corr_active = m2 * n_corr_acq + (m3 + m4) * n_corr_track + m5 * (0.75 * n_corr_acq)
    p_corr = gps.p_corr_per_w * n_corr_active * gps.f_clk_rel * (gps.vdd_v**2)

    # RF: IF bandwidth depends on tracking loop bandwidth
    b_if = 2.0 * (b_loop + gps.f_dop_max_hz)
    b_if = max(0.0, min(gps.b_if_max_hz, b_if))
    # NF(T) increases away from optimum; we fold it into a mild RF overhead
    nf_t = gps.nf0 + gps.delta_nf_per_c * (s.t_gps_c - gps.t_nf_opt_c)
    g_t = 1.0 + 0.02 * (nf_t - gps.nf0)
    p_if = gps.p_if0_w * (1.0 + gps.zeta_b_if * (b_if / max(1e-9, gps.b_if_max_hz))) * g_t
    # RF is active in ACQ/TRACK/NAV/ASSIST
    rf_on = (m2 + m3 + m4 + m5)
    p_rf = rf_on * (gps.p_lna_w + gps.p_mixer_w + p_if + gps.p_adc_w)

    # DSP: depends on f_update and required accuracy, also on satellites used
    n_sat_used = min(int(round(n_sat_locked)), gps.n_sat_optimal)
    acc_term = _log2(100.0 / sigma_req)  # tighter requirement -> larger term
    p_dsp = (gps.p_dsp_base_w + gps.k_dsp_w * f_update * (acc_term + gps.beta_sat * n_sat_used)) * (m4 + 0.25 * m3 + 0.30 * m5)

    # Memory access scales with update rate during NAV
    p_mem = (gps.p_mem0_w + gps.lambda_mem_access * gps.e_access_j * f_update) * (m4 + 0.20 * m3)

    # Clock (TCXO) cost (always-on small baseline when not OFF)
    p_clk = gps.p_clk0_w * (1.0 + gps.gamma_tcx0 * ((s.t_gps_c - gps.t_ref_c) ** 2)) * (gps.f_clk_rel / max(1e-9, gps.f_clk_ref))
    p_clk *= (1.0 - 0.40 * m0)  # OFF reduces clocking strongly

    # Leakage grows with temperature; always present when not OFF
    p_leak = gps.p_leak_25c_w * (math.e ** (gps.k_leak_per_c * (s.t_gps_c - 25.0)))
    p_leak *= (1.0 - 0.75 * m0)

    # Assisted mode uses extra network/processing (captured by p_dsp increase), plus a small assist overhead
    p_assist = 0.010 * assist * m5

    return max(0.0, p_rf + p_corr + p_dsp + p_mem + p_clk + p_leak + p_assist)


def gps_step(
    gps: GpsParams,
    s: GpsState,
    c: GpsControl,
    u: UsageInputs,
    t_substrate_c: float,
    t_shared_c: float,
    dt_s: float,
) -> Tuple[GpsState, float]:
    """Advance GPS CTMC + continuous states by one explicit-Euler step.

    Returns (next_state, p_gps_w).
    """

    # Clamp controls
    f_update = _clamp(c.f_update_hz, 0.0, 10.0)
    sigma_req = _clamp(c.sigma_pos_req_m, 1.0, 100.0)
    cn0_thresh = _clamp(c.cn0_thresh_dbhz, 20.0, 45.0)
    b_loop = _clamp(c.b_loop_hz, 1.0, 20.0)
    assist = _clamp(c.assist, 0.0, 1.0)

    # Enabled logic: either explicit gps_on or legacy duty/acq
    gps_on = 1.0 if clamp01(u.gps_on) >= 0.5 else 0.0
    if gps_on < 0.5:
        if u.gps_fix_duty > 0.0 or u.gps_acq_rate_hz > 0.0:
            gps_on = 1.0

    want = 1.0 if (gps_on >= 0.5 and f_update > 1e-6) else 0.0

    # Environmental inputs
    cn0_env = max(gps.cn0_min_dbhz, float(u.gps_cn0_env_dbhz))
    n_vis_raw = _clamp(float(u.gps_n_vis_raw), 0.0, 20.0)

    # N_vis low-pass
    n_vis_next = s.n_vis + ((n_vis_raw - s.n_vis) / max(1e-3, gps.tau_n_vis_s)) * dt_s
    n_vis_next = _clamp(n_vis_next, 0.0, 20.0)

    # NF(T) impacts effective CN0 equilibrium a bit
    nf_t = gps.nf0 + gps.delta_nf_per_c * (s.t_gps_c - gps.t_nf_opt_c)
    cn0_eq = cn0_env - 0.6 * (nf_t - gps.nf0)

    cn0_next = s.cn0_dbhz + (-gps.alpha_cn0 * (s.cn0_dbhz - cn0_eq)) * dt_s
    cn0_next = _clamp(cn0_next, gps.cn0_min_dbhz, 60.0)

    # Lock probability factor from CN0
    cn0_norm = _clamp((cn0_next - gps.cn0_ref_dbhz) / max(1e-9, gps.cn0_span_db), -2.0, 2.0)
    q_cn0 = _sigmoid(gps.k_cn0_lock * cn0_norm * 4.0)

    # LQ ideal: depends on CN0 threshold margin and locked sats
    lq_cn = _sigmoid(gps.k_lq_cn0 * (cn0_next - cn0_thresh))
    lq_sat = _sigmoid(gps.k_lq_sat * (s.n_locked - 4.0))
    lq_ideal = _clamp(lq_cn * lq_sat, 0.0, 1.0)
    lq_next = s.lq + ((lq_ideal - s.lq) / max(1e-3, gps.tau_lq_s)) * dt_s
    lq_next = _clamp(lq_next, 0.0, 1.0)

    # Acquisition progress: rises in ACQ/ASSIST, resets otherwise
    in_acq = (s.m2 + s.m5)
    k_acq = 0.35 + 0.65 * assist
    p_acq_next = s.p_acq + (in_acq * k_acq * q_cn0 * (1.0 - s.p_acq) - (1.0 - in_acq) * (s.p_acq / 1.2)) * dt_s
    p_acq_next = _clamp(p_acq_next, 0.0, 1.0)

    # Locked satellites dynamics
    lambda_lock = gps.lambda_lock_base * (0.4 + 0.6 * q_cn0) * (0.7 + 0.3 * assist)
    lambda_loss = gps.lambda_loss_base * (1.0 + 0.6 * (1.0 - lq_next))
    active_locking = (s.m2 + s.m3 + s.m4 + s.m5)
    n_locked_next = s.n_locked + (
        active_locking * lambda_lock * max(0.0, n_vis_next - s.n_locked)
        - (s.m3 + s.m4) * lambda_loss * s.n_locked * (1.0 - lq_next)
    ) * dt_s
    n_locked_next = _clamp(n_locked_next, 0.0, n_vis_next)

    # Accuracy dynamics: sigma floor improves with satellites and CN0, plus processing gain from tighter sigma_req
    import math

    sat_gain = math.sqrt(max(1.0, n_locked_next))
    cn0_gain = _clamp((cn0_next - gps.cn0_min_dbhz) / 20.0, 0.2, 2.0)
    sigma_floor = gps.sigma0_m / max(1e-6, sat_gain * cn0_gain)
    proc_gain = 1.0 + gps.k_proc_gain * _log2(100.0 / sigma_req)
    sigma_floor_eff = sigma_floor / max(1.0, proc_gain)

    # TCXO drift penalty
    sigma_tcxo = gps.k_sigma_tcxo_m * abs(s.t_gps_c - gps.t_ref_c)

    # Between updates, error grows; more frequent update reduces this term.
    update_term = gps.k_sigma_update_m / max(0.1, f_update) if f_update > 1e-9 else 50.0

    sigma_target_nav = sigma_floor_eff + sigma_tcxo
    sigma_target_track = sigma_floor_eff + sigma_tcxo + update_term

    w_nav = _clamp(s.m4, 0.0, 1.0)
    tau_sigma = w_nav * gps.tau_sigma_nav_s + (1.0 - w_nav) * gps.tau_sigma_track_s
    sigma_target = w_nav * sigma_target_nav + (1.0 - w_nav) * sigma_target_track
    sigma_next = s.sigma_pos_est_m + (-(s.sigma_pos_est_m - sigma_target) / max(1e-3, tau_sigma)) * dt_s
    sigma_next = _clamp(sigma_next, 0.5, 200.0)

    # CTMC transitions
    # Modes: 0 OFF, 1 STANDBY, 2 ACQ, 3 TRACK, 4 NAV, 5 ASSIST
    r = [[0.0 for _ in range(6)] for __ in range(6)]

    # Wake/sleep
    r[0][1] = want / max(1e-3, gps.tau_wake_s)
    r[1][0] = (1.0 - want) / max(1e-3, gps.tau_sleep_s)

    # Standby -> ACQ or ASSIST
    r[1][2] = (want * (1.0 - assist)) / max(1e-3, gps.tau_wake_s)
    r[1][5] = (want * assist) / max(1e-3, gps.tau_wake_s)

    # ACQ/ASSIST -> TRACK: depends on progress and satellites
    p_lock = _clamp(p_acq_next, 0.0, 1.0)
    r[2][3] = lambda_lock * (n_vis_next / 4.0) * p_lock
    r[5][3] = 1.8 * lambda_lock * (n_vis_next / 4.0) * p_lock

    # TRACK <-> NAV: periodic updates approximated as a rate into NAV and a short return time
    r[3][4] = want * f_update
    r[4][3] = 1.0 / max(1e-3, gps.tau_nav_s)

    # Signal loss to standby
    loss_rate = lambda_loss * math.exp(-_clamp(lq_next / max(1e-6, gps.lq0), 0.0, 10.0))
    r[3][1] = loss_rate
    r[4][1] = loss_rate

    # If GPS no longer wanted, wind down to standby
    r[2][1] += (1.0 - want) * 0.6
    r[5][1] += (1.0 - want) * 0.8
    r[3][1] += (1.0 - want) * 0.5
    r[4][1] += (1.0 - want) * 0.5

    m = [s.m0, s.m1, s.m2, s.m3, s.m4, s.m5]
    dm = [0.0 for _ in range(6)]
    for i in range(6):
        out_i = 0.0
        for j in range(6):
            if i == j:
                continue
            out_i += r[i][j]
            dm[j] += m[i] * r[i][j]
        dm[i] -= m[i] * out_i

    m_next = [max(0.0, m[i] + dm[i] * dt_s) for i in range(6)]
    ssum = sum(m_next)
    if ssum <= 1e-12:
        m_next = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ssum = 1.0
    m_next = [x / ssum for x in m_next]

    # Power and thermal
    tmp_state = GpsState(
        m0=m_next[0],
        m1=m_next[1],
        m2=m_next[2],
        m3=m_next[3],
        m4=m_next[4],
        m5=m_next[5],
        t_gps_c=s.t_gps_c,
        p_acq=p_acq_next,
        cn0_dbhz=cn0_next,
        sigma_pos_est_m=sigma_next,
        n_vis=n_vis_next,
        n_locked=n_locked_next,
        lq=lq_next,
    )
    p_gps = gps_expected_power_w(gps, tmp_state, c)

    dT_dt = (
        p_gps
        - (s.t_gps_c - t_substrate_c) / gps.r_th_c_per_w
        + gps.kappa_shared_w_per_c * (t_shared_c - s.t_gps_c)
    ) / gps.c_th_j_per_c
    t_gps_next = s.t_gps_c + dT_dt * dt_s

    return (
        GpsState(
            m0=m_next[0],
            m1=m_next[1],
            m2=m_next[2],
            m3=m_next[3],
            m4=m_next[4],
            m5=m_next[5],
            t_gps_c=t_gps_next,
            p_acq=p_acq_next,
            cn0_dbhz=cn0_next,
            sigma_pos_est_m=sigma_next,
            n_vis=n_vis_next,
            n_locked=n_locked_next,
            lq=lq_next,
        ),
        p_gps,
    )


def _sigmoid(x: float) -> float:
    import math

    # numerically stable-ish sigmoid
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def ble_rssi_inst_dbm(bt: BluetoothParams, state: BluetoothState) -> float:
    import math

    d = max(1e-3, bt.d_m)
    pl_db = bt.pl0_db + 10.0 * bt.pathloss_gamma * math.log10(d / max(1e-3, bt.d0_m))
    return bt.p_tx_peer_dbm - pl_db + state.x_s_db


def ble_per0(bt: BluetoothParams, rssi_dbm: float) -> float:
    # Higher RSSI -> smaller PER
    x = bt.per_sigmoid_a * (bt.rssi_ref_dbm - rssi_dbm)
    per0 = bt.per_min + (bt.per_max - bt.per_min) * _sigmoid(x)
    return max(bt.per_min, min(bt.per_max, per0))


def ble_eta_pa(bt: BluetoothParams, t_ble_c: float, p_tx_dbm: float) -> float:
    # η(T,P) = η0*[1 - 0.003(T-25)]*(P/P_ref)^0.1
    p_w = _dbm_to_w(p_tx_dbm)
    temp_factor = 1.0 - 0.003 * (t_ble_c - 25.0)
    temp_factor = max(0.7, min(1.1, temp_factor))
    p_factor = (max(1e-9, p_w) / max(1e-9, bt.p_ref_w)) ** 0.1
    eta = bt.eta0 * temp_factor * p_factor
    return max(0.05, min(0.6, eta))


def ble_capacity_bps(bt: BluetoothParams, state: BluetoothState) -> float:
    # Effective payload capacity when in a connection event
    eff = max(0.0, 1.0 - max(bt.per_min, min(bt.per_max, state.per)))
    return bt.r_phy_bps * eff


def ble_expected_power_w(bt: BluetoothParams, state: BluetoothState, ctrl: BluetoothControl) -> float:
    import math

    r_cmd = max(0.0, min(bt.r_max_bps, ctrl.r_cmd_bps))
    cap = ble_capacity_bps(bt, state)

    # Choose required event duration to support r_cmd on average:
    # r_cmd <= (tau_event/T_conn)*cap  => tau_event = r_cmd*T_conn/cap
    tau_event = 0.0 if cap <= 1e-9 else (r_cmd * bt.t_conn_s / cap)
    tau_event = max(0.0, min(bt.tau_event_max_s, tau_event))
    d_conn = 0.0 if bt.t_conn_s <= 1e-9 else max(0.0, min(1.0, tau_event / bt.t_conn_s))

    # Packet time proxy and payload bits per event
    active_time = max(0.0, tau_event - bt.tau_prep_s - bt.tau_turn_s)
    bits_per_event = min(bt.l_payload_max_bits, max(0.0, r_cmd * bt.t_conn_s))
    payload_time = min(active_time, bits_per_event / max(1e-9, bt.r_phy_bps))
    idle_time = max(0.0, active_time - payload_time)

    eta = ble_eta_pa(bt, state.t_ble_c, state.p_tx_dbm)
    p_tx_w = _dbm_to_w(state.p_tx_dbm)
    p_pkt_tx = p_tx_w / max(1e-9, eta)

    # RX term follows provided form: P_rx = base + K_rx*log2(1/PER)*(1+zeta*bw)
    per = max(bt.per_min, min(bt.per_max, state.per))
    rx_gain = math.log(1.0 / per, 2)
    p_pkt_rx = bt.p_rx_base_w + bt.k_rx * rx_gain * (1.0 + bt.zeta_bw * bt.b_eq_over_ref)

    # Average event power over tau_event
    if tau_event <= 1e-9:
        p_event_avg = 0.0
    else:
        p_event_avg = (
            bt.p_prepare_w * (bt.tau_prep_s / tau_event)
            + (p_pkt_tx + p_pkt_rx) * (payload_time / tau_event)
            + bt.p_idle_event_w * (idle_time / tau_event)
            + bt.p_idle_event_w * (bt.tau_turn_s / tau_event)
        )

    p_sleep = bt.p_sleep_25c_w * (1.0 + bt.k_sleep_per_c * (state.t_ble_c - 25.0))
    p_leak = bt.p_leak_25c_w * (2.718281828 ** (bt.k_leak_per_c * (state.t_ble_c - 25.0)))
    return d_conn * p_event_avg + (1.0 - d_conn) * p_sleep + p_leak


def ble_step(
    bt: BluetoothParams,
    state: BluetoothState,
    ctrl: BluetoothControl,
    u: UsageInputs,
    t_amb_c: float,
    t_adj_c: float,
    dt_s: float,
) -> Tuple[BluetoothState, float, float]:
    """Advance BLE continuous-time states.

    Returns (next_state, r_served_bps, p_ble_w).
    """

    import math

    # Fading OU (deterministic unless sigma_s_db>0)
    x_s_next = state.x_s_db + (-bt.omega_s * state.x_s_db) * dt_s

    # RSSI smoothing
    rssi_inst = ble_rssi_inst_dbm(bt, BluetoothState(
        t_ble_c=state.t_ble_c,
        e_accum_j=state.e_accum_j,
        per=state.per,
        rssi_avg_dbm=state.rssi_avg_dbm,
        x_s_db=x_s_next,
        p_tx_dbm=state.p_tx_dbm,
        q_bits=state.q_bits,
        n_event=state.n_event,
    ))
    tau_rssi = max(1e-3, bt.tau_rssi_s)
    rssi_avg_next = state.rssi_avg_dbm + ((rssi_inst - state.rssi_avg_dbm) / tau_rssi) * dt_s

    # PER dynamics toward PER0(RSSI)
    per0 = ble_per0(bt, rssi_inst)
    tau_per = max(1e-3, bt.tau_per_s)
    per_next = state.per + (-(state.per - per0) / tau_per) * dt_s
    per_next = max(bt.per_min, min(bt.per_max, per_next))

    # TX power control (optional adaptive)
    if bt.adaptive_power:
        per_err = (bt.per_target - per_next)
        # If PER is higher than target, per_err negative => increase power
        p_target = bt.p_tx_fixed_dbm - bt.alpha_per_dbm * per_err + bt.beta_rssi_dbm * (rssi_inst - rssi_avg_next)
    else:
        p_target = bt.p_tx_fixed_dbm

    p_target = max(bt.p_tx_min_dbm, min(bt.p_tx_max_dbm, p_target))
    tau_p = max(1e-3, bt.tau_p_tx_s)
    p_tx_dbm_next = state.p_tx_dbm + ((p_target - state.p_tx_dbm) / tau_p) * dt_s
    # quantize toward nearest level
    nearest = min(bt.p_tx_levels_dbm, key=lambda x: abs(x - p_tx_dbm_next))
    p_tx_dbm_next = nearest

    enabled = 1.0 if clamp01(u.bt_on) >= 0.5 else 0.0

    # Served rate is limited by being in events. We approximate duty from ctrl.
    r_cmd = enabled * max(0.0, min(bt.r_max_bps, ctrl.r_cmd_bps))
    cap = ble_capacity_bps(bt, BluetoothState(
        t_ble_c=state.t_ble_c,
        e_accum_j=state.e_accum_j,
        per=per_next,
        rssi_avg_dbm=rssi_avg_next,
        x_s_db=x_s_next,
        p_tx_dbm=p_tx_dbm_next,
        q_bits=state.q_bits,
        n_event=state.n_event,
    ))
    tau_event = 0.0 if cap <= 1e-9 else (r_cmd * bt.t_conn_s / cap)
    tau_event = max(0.0, min(bt.tau_event_max_s, tau_event))
    d_conn = 0.0 if bt.t_conn_s <= 1e-9 else max(0.0, min(1.0, tau_event / bt.t_conn_s))
    r_served = d_conn * min(r_cmd, cap)

    # Queue dynamics
    arrival = enabled * max(0.0, u.bt_arrival_bps)
    q_next = max(0.0, min(bt.q_max_bits, state.q_bits + (arrival - r_served) * dt_s))

    # Power and energy accumulation
    tmp_state = BluetoothState(
        t_ble_c=state.t_ble_c,
        e_accum_j=state.e_accum_j,
        per=per_next,
        rssi_avg_dbm=rssi_avg_next,
        x_s_db=x_s_next,
        p_tx_dbm=p_tx_dbm_next,
        q_bits=q_next,
        n_event=state.n_event,
    )
    p_ble = ble_expected_power_w(bt, tmp_state, ctrl)
    e_next = state.e_accum_j + p_ble * dt_s

    # Temperature dynamics
    dT_dt = (p_ble - (state.t_ble_c - t_amb_c) / bt.r_th_c_per_w + bt.kappa_adj_w_per_c * (t_adj_c - state.t_ble_c)) / bt.c_th_j_per_c
    t_next = state.t_ble_c + dT_dt * dt_s

    # Event counter (discrete)
    n_event_next = state.n_event + int(dt_s / max(1e-3, bt.t_conn_s))

    return (
        BluetoothState(
            t_ble_c=t_next,
            e_accum_j=e_next,
            per=per_next,
            rssi_avg_dbm=rssi_avg_next,
            x_s_db=x_s_next,
            p_tx_dbm=p_tx_dbm_next,
            q_bits=q_next,
            n_event=n_event_next,
        ),
        r_served,
        p_ble,
    )


def total_power_w(p: ComponentPowers, u: UsageInputs, temp_c: float) -> float:
    return sum(component_power_w(p, u, temp_c).values())


def dvfs_vdd(cpu: CpuParams, f_ghz: float) -> float:
    f = max(cpu.f_min_ghz, f_ghz)
    return cpu.v_min + cpu.dvfs_a * max(0.0, f - cpu.f_min_ghz) ** cpu.dvfs_beta


def f_max_throttled(cpu: CpuParams, t_j_c: float) -> float:
    if t_j_c <= cpu.t_throttle_c:
        return cpu.f_rated_ghz
    return cpu.f_rated_ghz * (1.0 - cpu.gamma_throttle_per_c * (t_j_c - cpu.t_throttle_c))


def cpu_power_components_w(cpu: CpuParams, state: CpuState, ctrl: CpuControl) -> Dict[str, float]:
    n = max(0, min(cpu.n_total, int(ctrl.n_active)))
    f = max(cpu.f_min_ghz, float(ctrl.f_ghz))
    v = dvfs_vdd(cpu, f)
    u = clamp01(state.u_eff)

    p_dyn = cpu.c_eff * n * f * (v**2) * u
    # Leakage with temperature exponent (use °C -> K)
    t_k = (state.t_j_c + 273.15)
    p_static = cpu.k_leak * n * v * (2.718281828 ** (cpu.kappa_leak * (t_k / cpu.t0_k - 1.0)))
    p_clk = cpu.c_clock * n * f * (v**2)

    return {"dynamic": p_dyn, "static": p_static, "clock": p_clk}


def cpu_total_power_w(cpu: CpuParams, state: CpuState, ctrl: CpuControl) -> float:
    comps = cpu_power_components_w(cpu, state, ctrl)
    return comps["dynamic"] + comps["static"] + comps["clock"]


def cpu_perf(cpu: CpuParams, state: CpuState, ctrl: CpuControl) -> float:
    """Normalized compute supply in [0, ~1]."""
    n = max(0, min(cpu.n_total, int(ctrl.n_active)))
    f = max(cpu.f_min_ghz, float(ctrl.f_ghz))
    u = clamp01(state.u_eff)
    supply = (n / cpu.n_total) * (f / cpu.f_rated_ghz) * u
    return cpu.perf_scale * supply


def cpu_step(
    cpu: CpuParams,
    state: CpuState,
    ctrl: CpuControl,
    t_amb_c: float,
    dt_s: float,
) -> CpuState:
    """One explicit step of CPU ODEs.

    u_eff dynamics:
      du_eff/dt = (u_cmd - u_eff)/tau_workload

    thermal:
      C_th dT/dt = P_total - (T-Tamb)/R_th - h(T-Tamb)

    thermal buffer proxy:
      dQ/dt = P_total - Q/tau_q
    """

    u_cmd = clamp01(ctrl.u_cmd)
    tau_u = max(1e-3, cpu.tau_workload_s)
    du_dt = (u_cmd - state.u_eff) / tau_u
    u_eff_next = clamp01(state.u_eff + du_dt * dt_s)

    # Apply throttling to control frequency if needed
    f_cap = max(cpu.f_min_ghz, f_max_throttled(cpu, state.t_j_c))
    f_next = min(max(cpu.f_min_ghz, ctrl.f_ghz), f_cap)
    ctrl_eff = CpuControl(f_ghz=f_next, n_active=int(ctrl.n_active), u_cmd=u_cmd)

    state_mid = CpuState(t_j_c=state.t_j_c, q_thermal_j=state.q_thermal_j, u_eff=u_eff_next)
    p_total = cpu_total_power_w(cpu, state_mid, ctrl_eff)

    dT_dt = (p_total - (state.t_j_c - t_amb_c) / cpu.r_th_c_per_w - cpu.h_cool_w_per_c * (state.t_j_c - t_amb_c)) / cpu.c_th_j_per_c
    t_next = state.t_j_c + dT_dt * dt_s

    dQ_dt = p_total - state.q_thermal_j / max(1e-3, cpu.tau_q_s)
    q_next = state.q_thermal_j + dQ_dt * dt_s

    return CpuState(t_j_c=t_next, q_thermal_j=q_next, u_eff=u_eff_next)


def cpu_step_workload_only(cpu: CpuParams, state: CpuState, ctrl: CpuControl, dt_s: float) -> Tuple[CpuState, CpuControl, float]:
    """CPU step used with the 3-node thermal network.

    Updates u_eff and q_thermal_j, but does NOT update temperature.

    Returns: (next_state, ctrl_eff, p_cpu_w)
    """

    u_cmd = clamp01(ctrl.u_cmd)
    tau_u = max(1e-3, cpu.tau_workload_s)
    du_dt = (u_cmd - state.u_eff) / tau_u
    u_eff_next = clamp01(state.u_eff + du_dt * dt_s)

    # Apply throttling to control frequency if needed
    f_cap = max(cpu.f_min_ghz, f_max_throttled(cpu, state.t_j_c))
    f_next = min(max(cpu.f_min_ghz, float(ctrl.f_ghz)), f_cap)
    ctrl_eff = CpuControl(f_ghz=f_next, n_active=int(ctrl.n_active), u_cmd=u_cmd)

    state_mid = CpuState(t_j_c=state.t_j_c, q_thermal_j=state.q_thermal_j, u_eff=u_eff_next)
    p_total = cpu_total_power_w(cpu, state_mid, ctrl_eff)

    dQ_dt = p_total - state.q_thermal_j / max(1e-3, cpu.tau_q_s)
    q_next = state.q_thermal_j + dQ_dt * dt_s

    return CpuState(t_j_c=state.t_j_c, q_thermal_j=q_next, u_eff=u_eff_next), ctrl_eff, p_total


def _db_to_lin(db: float) -> float:
    return 10.0 ** (db / 10.0)


def _lin_to_db(lin: float) -> float:
    if lin <= 1e-30:
        return -300.0
    import math

    return 10.0 * math.log10(lin)


def _log2(x: float) -> float:
    import math

    return math.log(x, 2)


def cellular_capacity_bps(cell: CellularParams, state: CellularState) -> float:
    n_w = cell.n0_w_per_hz * cell.nf_lin * cell.b_hz
    snr_lin = max(0.0, state.p_tx_w) / max(1e-15, n_w)
    return cell.b_hz * _log2(1.0 + snr_lin)


def _cellular_capacity_bps_for_bw(cell: CellularParams, state: CellularState, b_hz: float) -> float:
    b = max(1.0, float(b_hz))
    n_w = cell.n0_w_per_hz * cell.nf_lin * b
    snr_lin = max(0.0, state.p_tx_w) / max(1e-15, n_w)
    return b * _log2(1.0 + snr_lin)


def _mcs_order_from_spectral_eff(se: float) -> int:
    # Map bits/s/Hz requirement to modulation order log2(M)
    if se <= 1.2:
        return 2  # QPSK
    if se <= 3.0:
        return 4  # 16QAM
    if se <= 5.0:
        return 6  # 64QAM
    return 8      # 256QAM


def _eta_pa(cell: CellularParams, t_modem_c: float, p_tx_w: float) -> float:
    import math

    temp_factor = 1.0 - cell.kappa_t * (t_modem_c - cell.t_opt_c)
    temp_factor = max(0.3, min(1.2, temp_factor))
    tanh_term = math.tanh(cell.xi * max(0.0, p_tx_w) / max(1e-9, cell.p_tx_max_w))
    eta = cell.eta_max * temp_factor * max(0.05, tanh_term)
    return max(0.05, min(cell.eta_max, eta))


def _w_to_dbm(p_w: float) -> float:
    import math

    if p_w <= 1e-30:
        return -300.0
    return 10.0 * math.log10(p_w * 1000.0)


def _dbm_to_w(dbm: float) -> float:
    return 10.0 ** (dbm / 10.0) / 1000.0


def _eta_pa_wifi(wifi: WiFiParams, t_wifi_c: float, p_tx_w: float) -> float:
    import math

    temp_factor = 1.0 - wifi.kappa_t * (t_wifi_c - 25.0)
    temp_factor = max(0.3, min(1.2, temp_factor))
    tanh_term = 1.0 - math.exp(-wifi.xi * max(0.0, p_tx_w) / max(1e-9, wifi.p_sat_w))
    eta = wifi.eta0_pa * temp_factor * max(0.05, tanh_term)
    return max(0.05, min(wifi.eta0_pa, eta))


def wifi_capacity_bps(wifi: WiFiParams, state: WiFiState) -> float:
    import math

    # Received SNR based on pathloss + fading (in dB)
    pl_db = wifi.pl0_db - wifi.g_tx_db - wifi.g_rx_db - state.x_f_db
    loss_lin = 10.0 ** (pl_db / 10.0)
    p_rx_w = max(0.0, state.p_tx_w) / max(1e-12, loss_lin)

    n_w = wifi.n0_w_per_hz * wifi.nf_lin * wifi.bw_hz
    snr_lin = p_rx_w / max(1e-15, n_w)
    se = math.log(1.0 + max(0.0, snr_lin), 2)
    return max(0.0, float(wifi.n_streams)) * wifi.bw_hz * se


def wifi_power_components_w(
    wifi: WiFiParams,
    state: WiFiState,
    ctrl: WiFiControl,
) -> Dict[str, float]:
    import math

    r_cmd = max(0.0, min(wifi.r_max_bps, ctrl.r_cmd_bps))
    b = max(1.0, wifi.bw_hz)
    se_req = r_cmd / b

    # Modulation order proxy via spectral efficiency
    if se_req <= 1.2:
        log2m = 2.0
    elif se_req <= 3.0:
        log2m = 4.0
    elif se_req <= 5.0:
        log2m = 6.0
    else:
        log2m = 8.0

    eta_pa = _eta_pa_wifi(wifi, state.t_wifi_c, state.p_tx_w)
    p_pa = max(0.0, state.p_tx_w) / eta_pa

    p_bb = wifi.p_bb_base_w + wifi.k_bb_j_per_bit * r_cmd * max(1.0, float(wifi.n_streams)) * log2m
    p_rf = wifi.p_rf0_w * (1.0 + wifi.zeta_rf * (wifi.bw_hz / max(1.0, wifi.bw_max_hz))) * (1.0 + 0.002 * (state.t_wifi_c - 25.0))
    p_clk = wifi.p_clk0_w
    p_leak = wifi.p_leak_25c_w * (2.718281828 ** (wifi.k_leak_per_c * (state.t_wifi_c - 25.0)))

    return {"pa": p_pa, "bb": p_bb, "rf": p_rf, "clock": p_clk, "leak": p_leak}


def wifi_expected_power_w(
    wifi: WiFiParams,
    state: WiFiState,
    ctrl: WiFiControl,
) -> float:
    comps = wifi_power_components_w(wifi, state, ctrl)
    p_tx = comps["pa"] + comps["bb"] + comps["rf"] + comps["clock"] + comps["leak"]
    p_rx = comps["bb"] * 0.55 + comps["rf"] + comps["clock"] + comps["leak"]

    return (
        state.p1 * wifi.p_deep_sleep_w
        + state.p2 * wifi.p_light_sleep_w
        + state.p3 * wifi.p_idle_w
        + state.p4 * p_rx
        + state.p5 * p_tx
        + state.p6 * wifi.p_cca_w
    )


def wifi_step(
    wifi: WiFiParams,
    state: WiFiState,
    ctrl: WiFiControl,
    u: UsageInputs,
    t_amb_c: float,
    t_cpu_c: float,
    t_cell_c: float,
    dt_s: float,
) -> Tuple[WiFiState, float, float]:
    """Advance WiFi CTMC+queue+P_tx+thermal.

    Returns (next_state, r_served_bps, p_expected_w).
    """

    import math

    r_cmd = max(0.0, min(wifi.r_max_bps, ctrl.r_cmd_bps))
    arrival = max(0.0, u.wifi_rate_bps)

    # --- fading (OU, deterministic here) ---
    tau_f = max(1e-3, wifi.tau_f_s)
    x_f_next = state.x_f_db + (-state.x_f_db / tau_f) * dt_s

    # --- RSSI and TX power control ---
    # RSSI_meas(dBm) ~= P_tx(dBm) - PL(dB) + G + X_f
    pl_db = wifi.pl0_db
    rssi_meas_dbm = _w_to_dbm(state.p_tx_w) - pl_db + wifi.g_tx_db + wifi.g_rx_db + x_f_next
    rssi_err_db = max(0.0, wifi.rssi_target_dbm - rssi_meas_dbm)

    p_tx_target = wifi.p_tx_min_w + wifi.alpha_rssi_w_per_db * rssi_err_db
    if wifi.r_ref_bps > 1.0 and r_cmd > 0.0:
        p_tx_target += wifi.beta_rate_w * (r_cmd / wifi.r_ref_bps) ** max(0.1, wifi.gamma_rate)
    p_tx_target = max(wifi.p_tx_min_w, min(wifi.p_tx_max_w, p_tx_target))

    tau_p = max(1e-3, wifi.tau_p_tx_s)
    p_tx_next = state.p_tx_w + ((p_tx_target - state.p_tx_w) / tau_p) * dt_s
    p_tx_next = max(wifi.p_tx_min_w, min(wifi.p_tx_max_w, p_tx_next))

    # --- queue dynamics (TX only) ---
    cap = wifi_capacity_bps(
        wifi,
        WiFiState(
            p0=state.p0,
            p1=state.p1,
            p2=state.p2,
            p3=state.p3,
            p4=state.p4,
            p5=state.p5,
            p6=state.p6,
            p_tx_w=p_tx_next,
            t_wifi_c=state.t_wifi_c,
            q_wifi_bits=state.q_wifi_bits,
            x_f_db=x_f_next,
            tau_state_s=state.tau_state_s,
        ),
    )
    r_phy = min(r_cmd, cap)
    r_served = max(0.0, state.p5) * r_phy

    q_next = max(0.0, min(wifi.q_max_bits, state.q_wifi_bits + (arrival - r_served) * dt_s))

    # --- CTMC transition rates depend on queue/traffic/scan ---
    want_data = 1.0 if (q_next > wifi.q_wake_bits or arrival > 1.0) else 0.0
    traffic_level = 0.0 if wifi.q_max_bits <= 0 else min(1.0, q_next / wifi.q_max_bits)
    scan_busy = min(1.0, max(0.0, u.wifi_scan_rate_hz) * 0.6)

    # transitions (1/s)
    r12 = (1.0 / max(1e-3, wifi.tau_deep_wake_s)) * (0.2 + 0.8 * min(1.0, traffic_level + scan_busy))
    r21 = wifi.lambda_deeper_sleep * (1.0 - want_data)
    r23 = wifi.lambda_traffic * traffic_level * want_data
    r32 = wifi.lambda_sleep_down * (1.0 - want_data)
    r35 = wifi.lambda_tx_start * want_data * (0.2 + 0.8 * (r_cmd / max(1.0, wifi.r_max_bps)))
    r34 = wifi.lambda_rx_start * want_data * 0.25
    r36 = wifi.lambda_busy * scan_busy
    r53 = wifi.lambda_back_idle
    r43 = wifi.lambda_back_idle
    r63 = wifi.lambda_back_idle
    r45 = wifi.lambda_tx_rx_switch * 0.25
    r54 = wifi.lambda_tx_rx_switch * 0.25

    # probabilities
    p0, p1, p2, p3, p4, p5, p6 = state.p0, state.p1, state.p2, state.p3, state.p4, state.p5, state.p6

    # dp/dt = inflow - outflow
    dp0 = 0.0
    dp1 = -p1 * r12 + p2 * r21
    dp2 = p1 * r12 - p2 * (r21 + r23) + p3 * r32
    dp3 = p2 * r23 - p3 * (r32 + r34 + r35 + r36) + p4 * r43 + p5 * r53 + p6 * r63
    dp4 = p3 * r34 - p4 * (r43 + r45) + p5 * r54
    dp5 = p3 * r35 - p5 * (r53 + r54) + p4 * r45
    dp6 = p3 * r36 - p6 * r63

    p0n = max(0.0, p0 + dp0 * dt_s)
    p1n = max(0.0, p1 + dp1 * dt_s)
    p2n = max(0.0, p2 + dp2 * dt_s)
    p3n = max(0.0, p3 + dp3 * dt_s)
    p4n = max(0.0, p4 + dp4 * dt_s)
    p5n = max(0.0, p5 + dp5 * dt_s)
    p6n = max(0.0, p6 + dp6 * dt_s)

    s = p0n + p1n + p2n + p3n + p4n + p5n + p6n
    if s <= 1e-12:
        p0n, p1n, p2n, p3n, p4n, p5n, p6n = 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        p0n, p1n, p2n, p3n, p4n, p5n, p6n = (
            p0n / s,
            p1n / s,
            p2n / s,
            p3n / s,
            p4n / s,
            p5n / s,
            p6n / s,
        )

    # dominant state residence time proxy
    probs = [p0n, p1n, p2n, p3n, p4n, p5n, p6n]
    i_star = max(range(7), key=lambda i: probs[i])
    out_rate = 0.0
    if i_star == 1:
        out_rate = r12
    elif i_star == 2:
        out_rate = r21 + r23
    elif i_star == 3:
        out_rate = r32 + r34 + r35 + r36
    elif i_star == 4:
        out_rate = r43 + r45
    elif i_star == 5:
        out_rate = r53 + r54
    elif i_star == 6:
        out_rate = r63

    tau_next = state.tau_state_s + (1.0 - state.tau_state_s * out_rate) * dt_s
    if tau_next < 0.0:
        tau_next = 0.0

    next_state = WiFiState(
        p0=p0n,
        p1=p1n,
        p2=p2n,
        p3=p3n,
        p4=p4n,
        p5=p5n,
        p6=p6n,
        p_tx_w=p_tx_next,
        t_wifi_c=state.t_wifi_c,
        q_wifi_bits=q_next,
        x_f_db=x_f_next,
        tau_state_s=tau_next,
    )

    # --- thermal ---
    p_eff = wifi_expected_power_w(wifi, next_state, ctrl)
    dT_dt = (
        p_eff
        - (state.t_wifi_c - t_amb_c) / wifi.r_th_c_per_w
        + wifi.kappa_cpu_w_per_c * (t_cpu_c - state.t_wifi_c)
        + wifi.kappa_cell_w_per_c * (t_cell_c - state.t_wifi_c)
    ) / wifi.c_th_j_per_c
    t_wifi_next = state.t_wifi_c + dT_dt * dt_s

    next_state.t_wifi_c = t_wifi_next
    return next_state, r_served, p_eff


def cellular_power_w(
    cell: CellularParams,
    state: CellularState,
    ctrl: CellularControl,
    t_amb_c: float,
    t_cpu_c: float,
    *,
    b_hz: Optional[float] = None,
    rf_idle_fraction: float = 0.35,
) -> float:
    """Radio power in an active transfer state.

    Key modeling choice: treat the PA and a portion of RF as scaling with a
    transmit duty factor (average over time). This keeps low-throughput traffic
    from looking like continuous full-power transmission.
    """

    r_cmd = max(0.0, min(cell.r_max_bps, ctrl.r_cmd_bps))
    b = cell.b_hz if b_hz is None else max(1.0, float(b_hz))

    cap = _cellular_capacity_bps_for_bw(cell, state, b)
    tx_duty = 0.0 if cap <= 1e-9 else max(0.0, min(1.0, r_cmd / cap))

    se = 0.0 if b <= 0 else r_cmd / b
    mcs_order = _mcs_order_from_spectral_eff(se)

    eta_pa = _eta_pa(cell, state.t_modem_c, state.p_tx_w)
    p_pa = (max(0.0, state.p_tx_w) * tx_duty) / eta_pa

    # Baseband: base + per-bit energy scaled by modulation complexity (2^mcs_order)
    eta_bb = 1.0
    p_bb = cell.p_bb_base_w + cell.mu_bb_j_per_bit * r_cmd * (2**mcs_order) / eta_bb

    g_t = 1.0 + 0.002 * (state.t_modem_c - 25.0)
    p_rf_on = cell.p_rf0_w * (1.0 + cell.zeta_rf * (b / max(1.0, cell.b_max_hz))) * g_t
    rf_idle_fraction = max(0.0, min(1.0, float(rf_idle_fraction)))
    p_rf = p_rf_on * (rf_idle_fraction + (1.0 - rf_idle_fraction) * tx_duty)

    return p_pa + p_bb + p_rf


def cellular_expected_power_w(
    cell: CellularParams,
    state: CellularState,
    ctrl: CellularControl,
    t_amb_c: float,
    t_cpu_c: float,
) -> float:
    # Distinguish low-activity from high-activity states.
    # Low activity uses a narrower effective bandwidth and higher RF idle fraction.
    # Also cap the effective commanded rate in LOW state so that a high-rate command
    # doesn't unrealistically get "forced" through a narrowband low-activity mode.
    b_low = min(cell.b_hz, 5e6)
    ctrl_low = CellularControl(r_cmd_bps=min(max(0.0, ctrl.r_cmd_bps), cell.r_high_bps))
    p_low = cellular_power_w(
        cell,
        state,
        ctrl_low,
        t_amb_c=t_amb_c,
        t_cpu_c=t_cpu_c,
        b_hz=b_low,
        rf_idle_fraction=0.55,
    )
    p_high = cellular_power_w(
        cell,
        state,
        ctrl,
        t_amb_c=t_amb_c,
        t_cpu_c=t_cpu_c,
        b_hz=cell.b_hz,
        rf_idle_fraction=0.25,
    )

    return (
        state.p0 * cell.p_idle_w
        + state.p1 * cell.p_connected_w
        + state.p2 * p_low
        + state.p3 * p_high
        + state.p4 * cell.p_search_w
    )


def cellular_step(
    cell: CellularParams,
    state: CellularState,
    ctrl: CellularControl,
    u: UsageInputs,
    t_amb_c: float,
    t_cpu_c: float,
    dt_s: float,
) -> CellularState:
    import math

    # --- Queue dynamics ---
    cap = cellular_capacity_bps(cell, state)
    r_cmd = max(0.0, min(cell.r_max_bps, ctrl.r_cmd_bps))
    r_served = min(r_cmd, cap)
    dq_dt = max(0.0, u.cell_arrival_bps) - r_served
    q_next = max(0.0, state.q_data_bits + dq_dt * dt_s)

    # --- Path loss shadowing (OU, deterministic unless sigma_x_db>0 with external noise) ---
    x_next = state.x_sigma_db + (-cell.theta_x * state.x_sigma_db) * dt_s

    # --- TX power relaxes toward target implied by SNR_target ---
    # Simplify: SNR_meas depends on P_tx and (PL0 + Xsigma). Use a linear mapping:
    #   P_target = N * 10^(SNR_target/10) * 10^(PL/10)
    # Here PL is treated as dimensionless loss factor relative to PL0.
    pl_db = cell.pl0_db + x_next
    loss_lin = 10.0 ** (pl_db / 10.0)
    n_w = cell.n0_w_per_hz * cell.nf_lin * cell.b_hz
    p_target = n_w * (10.0 ** (cell.snr_target_db / 10.0)) * loss_lin
    p_target = max(cell.p_tx_min_w, min(cell.p_tx_max_w, p_target))

    tau_tx = max(1e-3, cell.tau_tx_s)
    dp_dt = (p_target - state.p_tx_w) / tau_tx
    p_tx_next = max(cell.p_tx_min_w, min(cell.p_tx_max_w, state.p_tx_w + dp_dt * dt_s))

    # --- CTMC transition rates depend on queue and commanded rate ---
    q = q_next
    # Treat nonzero offered load as keeping the modem in connected/active states
    want_data = 1.0 if (q > cell.q_wake_bits or u.cell_arrival_bps > 1.0) else 0.0
    high = 1.0 if r_cmd >= cell.r_high_bps else 0.0

    # rates (1/s)
    r01 = cell.lambda_connect * want_data  # IDLE->CONNECTED
    r04 = cell.lambda_search               # IDLE->SEARCH
    r10 = cell.lambda_timeout * (1.0 - want_data)  # CONNECTED->IDLE when no backlog
    r12 = cell.lambda_to_low * want_data * (1.0 - high)
    r13 = cell.lambda_to_high * want_data * high
    r20 = cell.lambda_timeout * (1.0 - want_data)
    r30 = cell.lambda_timeout * (1.0 - want_data)
    r40 = 1 / 8.0  # SEARCH->IDLE

    # dp/dt = Λ p (build from rates)
    p0, p1, p2, p3, p4 = state.p0, state.p1, state.p2, state.p3, state.p4

    dp0 = -p0 * (r01 + r04) + p1 * r10 + p2 * r20 + p3 * r30 + p4 * r40
    dp1 = p0 * r01 - p1 * (r10 + r12 + r13)
    dp2 = p1 * r12 - p2 * r20
    dp3 = p1 * r13 - p3 * r30
    dp4 = p0 * r04 - p4 * r40

    p0n = p0 + dp0 * dt_s
    p1n = p1 + dp1 * dt_s
    p2n = p2 + dp2 * dt_s
    p3n = p3 + dp3 * dt_s
    p4n = p4 + dp4 * dt_s

    # clamp and renormalize
    p0n = max(0.0, p0n)
    p1n = max(0.0, p1n)
    p2n = max(0.0, p2n)
    p3n = max(0.0, p3n)
    p4n = max(0.0, p4n)
    s = p0n + p1n + p2n + p3n + p4n
    if s <= 1e-12:
        p0n, p1n, p2n, p3n, p4n = 1.0, 0.0, 0.0, 0.0, 0.0
    else:
        p0n, p1n, p2n, p3n, p4n = p0n / s, p1n / s, p2n / s, p3n / s, p4n / s

    # --- Modem power and temperature ---
    tmp_state = CellularState(
        p0=p0n,
        p1=p1n,
        p2=p2n,
        p3=p3n,
        p4=p4n,
        p_tx_w=p_tx_next,
        t_modem_c=state.t_modem_c,
        q_data_bits=q_next,
        x_sigma_db=x_next,
    )
    p_eff = cellular_expected_power_w(cell, tmp_state, ctrl, t_amb_c=t_amb_c, t_cpu_c=t_cpu_c)

    dT_dt = (p_eff - (state.t_modem_c - t_amb_c) / cell.r_th_c_per_w + cell.kappa_cpu_w_per_c * (t_cpu_c - state.t_modem_c)) / cell.c_th_j_per_c
    t_modem_next = state.t_modem_c + dT_dt * dt_s

    return CellularState(
        p0=p0n,
        p1=p1n,
        p2=p2n,
        p3=p3n,
        p4=p4n,
        p_tx_w=p_tx_next,
        t_modem_c=t_modem_next,
        q_data_bits=q_next,
        x_sigma_db=x_next,
    )


def eta_conv(screen: ScreenParams, t_s_c: float) -> float:
    # η_conv = η0 * [1 - λ(Ts - Topt)]
    eta = screen.eta0_conv * (1.0 - screen.lambda_eta_per_c * (t_s_c - screen.t_opt_c))
    # avoid non-physical values
    if eta < 0.60:
        return 0.60
    if eta > 0.98:
        return 0.98
    return eta


def screen_power_components_w(
    screen: ScreenParams,
    u: UsageInputs,
    s: ScreenState,
) -> Dict[str, float]:
    """Compute P_backlight/P_driver/P_leakage using the reduced-order equations.

    Notes:
    - OLED emission: P_backlight ~ P_max * (L_eff^gamma) * exp(theta*(Ts-25))
      scaled by active area and resolution.
    - Driver: proportional to refresh rate, active area, resolution.
    - Leakage: exponential in temperature.
    """

    l_eff = clamp01(s.l_eff)
    gamma = max(0.2, u.screen_gamma)
    f_r = max(1.0, u.screen_refresh_hz)
    res_scale = clamp01(u.screen_res_scale)
    a_active = clamp01(u.screen_active_area)

    # Backlight / emission
    p_backlight = (
        screen.p_oled_max_w
        * (l_eff ** gamma)
        * (2.718281828 ** (screen.theta_per_c * (s.t_s_c - 25.0)))
        * a_active
        * res_scale
    )

    # Driver
    p_driver = screen.p_driver_60hz_w * (f_r / 60.0) * a_active * res_scale

    # Leakage
    p_leakage = screen.p_leak_25c_w * (2.718281828 ** (screen.k_leak_per_c * (s.t_s_c - 25.0)))

    return {
        "backlight": p_backlight,
        "driver": p_driver,
        "leakage": p_leakage,
    }


def screen_power_w(screen: ScreenParams, u: UsageInputs, s: ScreenState) -> float:
    comps = screen_power_components_w(screen, u, s)
    return comps["backlight"] + comps["driver"] + comps["leakage"]


def screen_step(
    screen: ScreenParams,
    s: ScreenState,
    u: UsageInputs,
    t_env_c: float,
    t_cpu_c: float,
    dt_s: float,
) -> ScreenState:
    """One explicit-Euler step of the screen ODEs.

    State vector x = [Ts, Q_pixel, L_eff]^T
      dL_eff/dt = (L_cmd - L_eff)/tau_response
      dTs/dt    = (1/C_th){P_screen - (Ts-Tenv)/R_th + κ(T_cpu - Ts)}
      dQ/dt     = k_q * f_r * L_eff - Q/tau_q   (proxy for refresh switching charge)
    """

    l_cmd = clamp01(u.brightness)
    tau = max(1e-3, screen.tau_response_s)
    dL_dt = (l_cmd - s.l_eff) / tau
    l_eff_next = clamp01(s.l_eff + dL_dt * dt_s)

    # Use intermediate L_eff for power calc this step
    s_mid = ScreenState(t_s_c=s.t_s_c, q_pixel=s.q_pixel, l_eff=l_eff_next)
    p_scr = screen_power_w(screen, u, s_mid)

    dT_dt = (
        p_scr
        - (s.t_s_c - t_env_c) / screen.r_th_c_per_w
        + screen.kappa_cpu_w_per_c * (t_cpu_c - s.t_s_c)
    ) / screen.c_th_j_per_c
    t_s_next = s.t_s_c + dT_dt * dt_s

    f_r = max(1.0, u.screen_refresh_hz)
    dQ_dt = screen.k_q * f_r * l_eff_next - s.q_pixel / max(1e-3, screen.tau_q_s)
    q_next = s.q_pixel + dQ_dt * dt_s

    return ScreenState(t_s_c=t_s_next, q_pixel=q_next, l_eff=l_eff_next)


def thermal_step(
    thermal: ThermalParams,
    temp_c: float,
    p_sys_w: float,
    t_amb_c: float,
    dt_s: float,
) -> float:
    # C dT/dt = P - (T - Tamb)/R
    dtemp_dt = (p_sys_w - (temp_c - t_amb_c) / thermal.r_th_c_per_w) / thermal.c_th_j_per_c
    return temp_c + dtemp_dt * dt_s


def soc_step(
    battery: BatteryParams,
    soc: float,
    p_sys_w: float,
    temp_c: float,
    dt_s: float,
) -> float:
    v = voc(battery, soc)
    i_bat = p_sys_w / (battery.eta_pmu * v)
    c_ah = c_eff_ah(battery, temp_c)
    dsoc_dt = -(i_bat + battery.i_sd_a) / (c_ah * 3600.0)
    return clamp01(soc + dsoc_dt * dt_s)


def simulate_soc(
    battery: BatteryParams,
    thermal: ThermalParams,
    powers: ComponentPowers,
    screen: ScreenParams,
    cpu: CpuParams,
    wifi: WiFiParams,
    bluetooth: BluetoothParams,
    gps: Optional[GpsParams],
    speaker: Optional[SpeakerParams],
    cellular: CellularParams,
    usage_fn: Callable[[float], UsageInputs],
    z0: float,
    t_amb_c: float,
    t0_c: float = 25.0,
    z_min: float = 0.02,
    dt_s: float = 1.0,
    t_max_s: float = 7 * 24 * 3600,
    battery_soc: Optional[BatterySocParams] = None,
    thermal_net: Optional[ThermalNetworkParams] = None,
    thermal_controller: Optional[Callable[[float, ThermalNetworkState, float, float, float, float, float], ThermalControl]] = None,
    screen_brightness_controller: Optional[
        Callable[[float, UsageInputs, ScreenState, float, float, float], float]
    ] = None,
    cpu_controller: Optional[Callable[[float, UsageInputs, CpuState, float, float], CpuControl]] = None,
    wifi_controller: Optional[
        Callable[[float, UsageInputs, WiFiState, float, float, float, float], WiFiControl]
    ] = None,
    bluetooth_controller: Optional[
        Callable[[float, UsageInputs, BluetoothState, float, float, float], BluetoothControl]
    ] = None,
    gps_controller: Optional[
        Callable[[float, UsageInputs, GpsState, float, float, float], GpsControl]
    ] = None,
    speaker_controller: Optional[
        Callable[[float, UsageInputs, SpeakerState, float, float, float, float], SpeakerControl]
    ] = None,
    cellular_controller: Optional[
        Callable[[float, UsageInputs, CellularState, float, float, float], CellularControl]
    ] = None,
) -> Tuple[float, Dict[str, list]]:
    """Return (t_empty_s, traces). t_empty_s==t_max_s if not emptied."""

    t = 0.0
    z = clamp01(z0)
    temp_c = t0_c
    bat_state = BatterySocState(
        t_bat_c=t0_c,
        q_max_ah=(battery_soc.q_nom_ah if battery_soc else battery.c_nom_ah),
        soc_avg=clamp01(z0),
    )

    therm_state = None
    therm_ctrl = ThermalControl(p_heat_w=0.0, p_heat_elec_w=0.0, throttle_factor=1.0)
    if thermal_net is not None:
        therm_state = ThermalNetworkState(t_cpu_c=t0_c, t_bat_c=t0_c, t_case_c=t0_c)
        temp_c = therm_state.t_case_c
    screen_state = ScreenState(t_s_c=t0_c, q_pixel=0.0, l_eff=0.0)
    cpu_state = CpuState(t_j_c=t0_c, q_thermal_j=0.0, u_eff=0.0)
    wifi_state = WiFiState(t_wifi_c=t0_c)
    bt_state = BluetoothState(t_ble_c=t0_c)
    gps_state = GpsState(t_gps_c=t0_c)
    spk_state = SpeakerState(t_vc_c=t0_c)
    cell_state = CellularState(t_modem_c=t0_c)

    traces: Dict[str, list] = {
        "t_s": [],
        "soc": [],
        "soc_pct": [],
        "temp_c": [],
        "t_case_c": [],
        "t_bat_c": [],
        "q_max_ah": [],
        "v_oc_v": [],
        "i_bat_a": [],
        "r_bat_ohm": [],
        "p_device_w": [],
        "p_loss_w": [],
        "p_heat_w": [],
        "p_heat_elec_w": [],
        "throttle_factor": [],
        "t_screen_c": [],
        "t_cpu_c": [],
        "p_sys_w": [],
        "p_screen_w": [],
        "p_cpu_w": [],
        "p_wifi_w": [],
        "p_bt_w": [],
        "p_gps_w": [],
        "p_spk_w": [],
        "p_cell_w": [],
        "l_req": [],
        "l_cmd": [],
        "l_eff": [],
        "cpu_demand": [],
        "cpu_f_ghz": [],
        "cpu_n_active": [],
        "cpu_u_eff": [],
        "cpu_u_cmd": [],

        "wifi_r_cmd_bps": [],
        "wifi_r_served_bps": [],
        "wifi_q_bits": [],
        "wifi_t_c": [],
        "wifi_p_tx_w": [],
        "wifi_p_tx_state": [],

        "bt_r_cmd_bps": [],
        "bt_r_served_bps": [],
        "bt_q_bits": [],
        "bt_t_c": [],
        "bt_per": [],
        "bt_rssi_dbm": [],
        "bt_p_tx_dbm": [],

        "gps_f_update_hz": [],
        "gps_sigma_req_m": [],
        "gps_sigma_est_m": [],
        "gps_cn0_dbhz": [],
        "gps_n_locked": [],
        "gps_lq": [],
        "gps_t_c": [],
        "gps_m_off": [],
        "gps_m_standby": [],
        "gps_m_acq": [],
        "gps_m_track": [],
        "gps_m_nav": [],
        "gps_m_assist": [],

        "spk_g_cmd": [],
        "spk_i_vc_a": [],
        "spk_v_filter_v": [],
        "spk_t_vc_c": [],
        "spk_p_joule_w": [],
        "spk_loud_est": [],

        "cell_r_cmd_bps": [],
        "cell_r_served_bps": [],
        "cell_q_bits": [],
        "cell_t_modem_c": [],
        "cell_p_tx_w": [],
        "cell_p_high": [],
    }

    while t < t_max_s and z > z_min:
        u_req = usage_fn(t)

        # Apply last-step thermal throttling to user-level demands (affects other components).
        if thermal_net is not None:
            tf = _clamp(therm_ctrl.throttle_factor, 0.0, 1.0)
            u_req = UsageInputs(
                brightness=clamp01(u_req.brightness * tf),
                cpu_demand=clamp01(u_req.cpu_demand * tf),
                screen_gamma=u_req.screen_gamma,
                screen_refresh_hz=u_req.screen_refresh_hz,
                screen_res_scale=u_req.screen_res_scale,
                screen_active_area=u_req.screen_active_area,
                gps_fix_duty=u_req.gps_fix_duty,
                gps_acq_rate_hz=u_req.gps_acq_rate_hz,
                wifi_rate_bps=max(0.0, u_req.wifi_rate_bps * tf),
                wifi_scan_rate_hz=u_req.wifi_scan_rate_hz,
                cell_arrival_bps=max(0.0, u_req.cell_arrival_bps * tf),
                bt_arrival_bps=max(0.0, u_req.bt_arrival_bps * tf),
                bt_on=u_req.bt_on,
                speaker_volume=u_req.speaker_volume,
                gps_update_min_hz=max(0.0, u_req.gps_update_min_hz * tf),
                gps_sigma_max_m=u_req.gps_sigma_max_m,
                gps_on=u_req.gps_on,
                gps_cn0_env_dbhz=u_req.gps_cn0_env_dbhz,
                gps_n_vis_raw=u_req.gps_n_vis_raw,
                gps_cn0_thresh_dbhz=u_req.gps_cn0_thresh_dbhz,
                gps_b_loop_hz=u_req.gps_b_loop_hz,
                gps_assist=u_req.gps_assist,
                spk_audio_level=getattr(u_req, "spk_audio_level", 0.0),
                spk_v_limit_v=getattr(u_req, "spk_v_limit_v", 1.0),
                spk_mode=getattr(u_req, "spk_mode", 0),
            )

        if battery_soc is None:
            v_bat = voc(battery, z)
        else:
            v_bat = battery_ocv_v(battery_soc, z)

        # Optional optimal control: override brightness command while keeping the
        # rest of scenario inputs unchanged.
        if screen_brightness_controller is None:
            l_cmd = u_req.brightness
        else:
            l_cmd = screen_brightness_controller(
                t,
                u_req,
                screen_state,
                t_amb_c,
                temp_c,
                dt_s,
            )

        u = UsageInputs(
            brightness=clamp01(l_cmd),
            cpu_demand=u_req.cpu_demand,

            screen_gamma=u_req.screen_gamma,
            screen_refresh_hz=u_req.screen_refresh_hz,
            screen_res_scale=u_req.screen_res_scale,
            screen_active_area=u_req.screen_active_area,

            gps_fix_duty=u_req.gps_fix_duty,
            gps_acq_rate_hz=u_req.gps_acq_rate_hz,

            gps_update_min_hz=u_req.gps_update_min_hz,
            gps_sigma_max_m=u_req.gps_sigma_max_m,
            gps_on=u_req.gps_on,
            gps_cn0_env_dbhz=u_req.gps_cn0_env_dbhz,
            gps_n_vis_raw=u_req.gps_n_vis_raw,
            gps_cn0_thresh_dbhz=u_req.gps_cn0_thresh_dbhz,
            gps_b_loop_hz=u_req.gps_b_loop_hz,
            gps_assist=u_req.gps_assist,

            wifi_rate_bps=u_req.wifi_rate_bps,
            wifi_scan_rate_hz=u_req.wifi_scan_rate_hz,
            cell_arrival_bps=u_req.cell_arrival_bps,
            bt_arrival_bps=u_req.bt_arrival_bps,
            bt_on=u_req.bt_on,
            speaker_volume=u_req.speaker_volume,

            spk_audio_level=getattr(u_req, "spk_audio_level", 0.0),
            spk_v_limit_v=getattr(u_req, "spk_v_limit_v", 1.0),
            spk_mode=getattr(u_req, "spk_mode", 0),
        )

        # CPU control (frequency, active cores, utilization command)
        if cpu_controller is None:
            # Baseline policy: medium frequency, half cores, choose u_cmd to meet demand if possible
            f0 = 0.7 * cpu.f_rated_ghz
            n0 = max(1, cpu.n_total // 2)
            denom = (n0 / cpu.n_total) * (f0 / cpu.f_rated_ghz)
            u_cmd0 = 0.0 if denom <= 1e-9 else clamp01(u.cpu_demand / denom)
            cpu_ctrl = CpuControl(f_ghz=f0, n_active=n0, u_cmd=u_cmd0)
        else:
            cpu_ctrl = cpu_controller(t, u, cpu_state, t_amb_c, dt_s)

        if thermal_net is None:
            cpu_state = cpu_step(cpu, cpu_state, cpu_ctrl, t_amb_c=t_amb_c, dt_s=dt_s)
            p_cpu = cpu_total_power_w(cpu, cpu_state, cpu_ctrl)
        else:
            cpu_state, cpu_ctrl, p_cpu = cpu_step_workload_only(cpu, cpu_state, cpu_ctrl, dt_s=dt_s)

        # WiFi control (data rate)
        if wifi_controller is None:
            r_req = max(0.0, u.wifi_rate_bps) + wifi_state.q_wifi_bits / max(1e-3, wifi.target_queue_delay_s)
            r_cmd = min(wifi.r_max_bps, r_req)
            wifi_ctrl = WiFiControl(r_cmd_bps=r_cmd)
        else:
            wifi_ctrl = wifi_controller(t, u, wifi_state, t_amb_c, cpu_state.t_j_c, cell_state.t_modem_c, dt_s)

        wifi_state, wifi_r_served, p_wifi = wifi_step(
            wifi=wifi,
            state=wifi_state,
            ctrl=wifi_ctrl,
            u=u,
            t_amb_c=t_amb_c,
            t_cpu_c=cpu_state.t_j_c,
            t_cell_c=cell_state.t_modem_c,
            dt_s=dt_s,
        )

        # Bluetooth control (data rate)
        if bluetooth_controller is None:
            enabled = 1.0 if clamp01(u.bt_on) >= 0.5 else 0.0
            r_req = enabled * (max(0.0, u.bt_arrival_bps) + bt_state.q_bits / max(1e-3, bluetooth.target_queue_delay_s))
            bt_ctrl = BluetoothControl(r_cmd_bps=min(bluetooth.r_max_bps, r_req))
        else:
            bt_ctrl = bluetooth_controller(t, u, bt_state, t_amb_c, temp_c, dt_s)

        bt_state, bt_r_served, p_bt = ble_step(
            bt=bluetooth,
            state=bt_state,
            ctrl=bt_ctrl,
            u=u,
            t_amb_c=t_amb_c,
            t_adj_c=(temp_c if thermal_net is None else float(therm_state.t_case_c)),
            dt_s=dt_s,
        )

        # GPS control (update rate + accuracy requirement)
        if gps is None:
            gps_ctrl = None
            p_gps = 0.0
        else:
            # Baseline: meet the minimum update rate if GPS is on; relax accuracy requirement to the max allowed.
            gps_enabled = 1.0 if clamp01(u.gps_on) >= 0.5 else 0.0
            if gps_enabled < 0.5 and (u.gps_fix_duty > 0.0 or u.gps_acq_rate_hz > 0.0):
                gps_enabled = 1.0

            if gps_controller is None:
                f_cmd = gps_enabled * _clamp(u.gps_update_min_hz, 0.0, 10.0)
                sigma_cmd = _clamp(u.gps_sigma_max_m, 1.0, 100.0)
                gps_ctrl = GpsControl(
                    sigma_pos_req_m=sigma_cmd,
                    f_update_hz=f_cmd,
                    cn0_thresh_dbhz=_clamp(u.gps_cn0_thresh_dbhz, 20.0, 45.0),
                    b_loop_hz=_clamp(u.gps_b_loop_hz, 1.0, 20.0),
                    assist=_clamp(u.gps_assist, 0.0, 1.0),
                )
            else:
                gps_ctrl = gps_controller(t, u, gps_state, t_amb_c, temp_c, dt_s)

            gps_state, p_gps = gps_step(
                gps=gps,
                s=gps_state,
                c=gps_ctrl,
                u=u,
                t_substrate_c=(temp_c if thermal_net is None else float(therm_state.t_case_c)),
                t_shared_c=cpu_state.t_j_c,
                dt_s=dt_s,
            )

        # Speaker control (volume)
        if speaker is None:
            spk_ctrl = None
            p_spk = 0.0
            p_joule = 0.0
            loud = 0.0
        else:
            if speaker_controller is None:
                spk_ctrl = SpeakerControl(
                    g=clamp01(u.speaker_volume),
                    v_limit_v=float(getattr(u, "spk_v_limit_v", 1.0)),
                    f_mode=int(getattr(u, "spk_mode", 0)),
                )
            else:
                spk_ctrl = speaker_controller(t, u, spk_state, t_amb_c, (temp_c if thermal_net is None else float(therm_state.t_case_c)), cpu_state.t_j_c, v_bat)

            spk_state, p_spk, p_joule, loud = speaker_step(
                spk=speaker,
                s=spk_state,
                c=spk_ctrl,
                u=u,
                t_amb_c=t_amb_c,
                t_phone_c=(temp_c if thermal_net is None else float(therm_state.t_case_c)),
                v_bat_v=v_bat,
                dt_s=dt_s,
            )

        # Cellular control (data rate)
        if cellular_controller is None:
            # Baseline: serve as much as possible, bounded by max
            r_cmd = min(cellular.r_max_bps, max(0.0, u.cell_arrival_bps))
            cell_ctrl = CellularControl(r_cmd_bps=r_cmd)
        else:
            cell_ctrl = cellular_controller(t, u, cell_state, t_amb_c, cpu_state.t_j_c, dt_s)

        cell_state = cellular_step(cellular, cell_state, cell_ctrl, u, t_amb_c=t_amb_c, t_cpu_c=cpu_state.t_j_c, dt_s=dt_s)
        cap = cellular_capacity_bps(cellular, cell_state)
        r_served = min(cell_ctrl.r_cmd_bps, cap)
        p_cell = cellular_expected_power_w(cellular, cell_state, cell_ctrl, t_amb_c=t_amb_c, t_cpu_c=cpu_state.t_j_c)

        # Step screen (uses CPU junction temp as proxy for T_cpu)
        screen_state = screen_step(
            screen=screen,
            s=screen_state,
            u=u,
            t_env_c=t_amb_c,
            t_cpu_c=cpu_state.t_j_c,
            dt_s=dt_s,
        )
        p_screen = screen_power_w(screen, u, screen_state)

        # Other components use the lumped temp (temp_c)
        other = component_power_w(powers, u, (temp_c if thermal_net is None else float(therm_state.t_case_c)))
        p_other = sum(v for k, v in other.items() if k not in ("wifi", "bt", "gps", "speaker"))

        # Battery loss model uses device power excluding internal loss.
        p_device = p_other + p_screen + p_cpu + p_wifi + p_bt + p_gps + p_spk + p_cell

        if battery_soc is None:
            p_loss = 0.0
            i_bat = p_device / max(1e-6, v_bat)
            r_bat = 0.0
            v_oc = v_bat
            p_sys = p_device
        else:
            t_bat_for_r = (bat_state.t_bat_c if thermal_net is None else float(therm_state.t_bat_c))
            p_loss, i_bat, v_oc, r_bat = battery_internal_loss_w(battery_soc, z, p_device, t_bat_c=t_bat_for_r)
            p_sys = p_device + p_loss

        # Thermal management (heater + throttling). Heater affects battery drain and battery heat.
        if thermal_net is not None and therm_state is not None and thermal_controller is not None:
            therm_ctrl = thermal_controller(
                t,
                therm_state,
                z,
                t_amb_c,
                p_cpu,
                p_loss,
                dt_s,
            )
        else:
            therm_ctrl = ThermalControl(p_heat_w=0.0, p_heat_elec_w=0.0, throttle_factor=1.0)

        if therm_ctrl.p_heat_elec_w > 0.0:
            p_device2 = p_device + max(0.0, therm_ctrl.p_heat_elec_w)
            if battery_soc is None:
                p_loss2 = 0.0
                i_bat2 = p_device2 / max(1e-6, v_bat)
                r_bat2 = 0.0
                v_oc2 = v_bat
                p_sys2 = p_device2
            else:
                t_bat_for_r = (bat_state.t_bat_c if thermal_net is None else float(therm_state.t_bat_c))
                p_loss2, i_bat2, v_oc2, r_bat2 = battery_internal_loss_w(battery_soc, z, p_device2, t_bat_c=t_bat_for_r)
                p_sys2 = p_device2 + p_loss2

            # Replace with the heater-including values
            p_device, p_loss, i_bat, r_bat, v_oc, p_sys = p_device2, p_loss2, i_bat2, r_bat2, v_oc2, p_sys2

        traces["t_s"].append(t)
        traces["soc"].append(z)
        traces["soc_pct"].append(100.0 * z)
        traces["temp_c"].append(temp_c if thermal_net is None else float(therm_state.t_case_c))
        traces["t_case_c"].append(temp_c if thermal_net is None else float(therm_state.t_case_c))
        traces["t_bat_c"].append(float(bat_state.t_bat_c if thermal_net is None else float(therm_state.t_bat_c)))
        traces["q_max_ah"].append(float(bat_state.q_max_ah))
        traces["v_oc_v"].append(float(v_oc))
        traces["i_bat_a"].append(float(i_bat))
        traces["r_bat_ohm"].append(float(r_bat))
        traces["p_device_w"].append(float(p_device))
        traces["p_loss_w"].append(float(p_loss))
        traces["p_heat_w"].append(float(therm_ctrl.p_heat_w))
        traces["p_heat_elec_w"].append(float(therm_ctrl.p_heat_elec_w))
        traces["throttle_factor"].append(float(therm_ctrl.throttle_factor))
        traces["t_screen_c"].append(screen_state.t_s_c)
        traces["t_cpu_c"].append(cpu_state.t_j_c)
        traces["p_sys_w"].append(p_sys)
        traces["p_screen_w"].append(p_screen)
        traces["p_cpu_w"].append(p_cpu)
        traces["p_wifi_w"].append(p_wifi)
        traces["p_bt_w"].append(p_bt)
        traces["p_gps_w"].append(p_gps)
        traces["p_spk_w"].append(p_spk)
        traces["p_cell_w"].append(p_cell)
        traces["l_req"].append(clamp01(u_req.brightness))
        traces["l_cmd"].append(clamp01(u.brightness))
        traces["l_eff"].append(clamp01(screen_state.l_eff))
        traces["cpu_demand"].append(clamp01(u.cpu_demand))
        traces["cpu_f_ghz"].append(float(cpu_ctrl.f_ghz))
        traces["cpu_n_active"].append(int(cpu_ctrl.n_active))
        traces["cpu_u_eff"].append(clamp01(cpu_state.u_eff))
        traces["cpu_u_cmd"].append(clamp01(cpu_ctrl.u_cmd))

        traces["wifi_r_cmd_bps"].append(float(wifi_ctrl.r_cmd_bps))
        traces["wifi_r_served_bps"].append(float(wifi_r_served))
        traces["wifi_q_bits"].append(float(wifi_state.q_wifi_bits))
        traces["wifi_t_c"].append(float(wifi_state.t_wifi_c))
        traces["wifi_p_tx_w"].append(float(wifi_state.p_tx_w))
        traces["wifi_p_tx_state"].append(float(wifi_state.p5))

        traces["bt_r_cmd_bps"].append(float(bt_ctrl.r_cmd_bps))
        traces["bt_r_served_bps"].append(float(bt_r_served))
        traces["bt_q_bits"].append(float(bt_state.q_bits))
        traces["bt_t_c"].append(float(bt_state.t_ble_c))
        traces["bt_per"].append(float(bt_state.per))
        traces["bt_rssi_dbm"].append(float(bt_state.rssi_avg_dbm))
        traces["bt_p_tx_dbm"].append(float(bt_state.p_tx_dbm))

        if gps is None or gps_ctrl is None:
            traces["gps_f_update_hz"].append(0.0)
            traces["gps_sigma_req_m"].append(100.0)
        else:
            traces["gps_f_update_hz"].append(float(gps_ctrl.f_update_hz))
            traces["gps_sigma_req_m"].append(float(gps_ctrl.sigma_pos_req_m))

        traces["gps_sigma_est_m"].append(float(gps_state.sigma_pos_est_m))
        traces["gps_cn0_dbhz"].append(float(gps_state.cn0_dbhz))
        traces["gps_n_locked"].append(float(gps_state.n_locked))
        traces["gps_lq"].append(float(gps_state.lq))
        traces["gps_t_c"].append(float(gps_state.t_gps_c))
        traces["gps_m_off"].append(float(gps_state.m0))
        traces["gps_m_standby"].append(float(gps_state.m1))
        traces["gps_m_acq"].append(float(gps_state.m2))
        traces["gps_m_track"].append(float(gps_state.m3))
        traces["gps_m_nav"].append(float(gps_state.m4))
        traces["gps_m_assist"].append(float(gps_state.m5))

        if speaker is None or spk_ctrl is None:
            traces["spk_g_cmd"].append(0.0)
        else:
            traces["spk_g_cmd"].append(float(spk_ctrl.g))
        traces["spk_i_vc_a"].append(float(spk_state.i_vc_a))
        traces["spk_v_filter_v"].append(float(spk_state.v_filter_v))
        traces["spk_t_vc_c"].append(float(spk_state.t_vc_c))
        traces["spk_p_joule_w"].append(float(p_joule))
        traces["spk_loud_est"].append(float(loud))

        traces["cell_r_cmd_bps"].append(float(cell_ctrl.r_cmd_bps))
        traces["cell_r_served_bps"].append(float(r_served))
        traces["cell_q_bits"].append(float(cell_state.q_data_bits))
        traces["cell_t_modem_c"].append(float(cell_state.t_modem_c))
        traces["cell_p_tx_w"].append(float(cell_state.p_tx_w))
        traces["cell_p_high"].append(float(cell_state.p3))

        if thermal_net is None:
            temp_c = thermal_step(thermal, temp_c, p_sys, t_amb_c, dt_s)

            if battery_soc is None:
                z = soc_step(battery, z, p_sys, temp_c, dt_s)
            else:
                # Battery temperature from user's algebraic model (smoothed)
                t_eq = battery_t_bat_eq_c(battery_soc, t_amb_c, p_cpu_w=p_cpu, p_bat_w=p_sys)
                tau_t = max(1e-3, battery_soc.tau_t_bat_s)
                t_bat_next = bat_state.t_bat_c + ((t_eq - bat_state.t_bat_c) / tau_t) * dt_s

                # Aging update (uses SOC_avg and T_bat)
                bat_state = BatterySocState(t_bat_c=t_bat_next, q_max_ah=bat_state.q_max_ah, soc_avg=bat_state.soc_avg)
                bat_state = battery_aging_step(battery_soc, bat_state, soc_frac=z, t_bat_c=t_bat_next, dt_s=dt_s)

                # SOC dynamics from battery current and available capacity
                q_ah = max(1e-9, bat_state.q_max_ah)
                i_total = max(0.0, i_bat) + max(0.0, battery.i_sd_a)
                dz_dt = -i_total / (q_ah * 3600.0)
                z = clamp01(z + dz_dt * dt_s)
        else:
            # 3-node thermal network update
            p_bat_heat = max(0.0, p_loss)
            therm_state = thermal_network_step(
                thermal_net,
                therm_state,
                t_env_c=t_amb_c,
                p_cpu_w=p_cpu,
                p_bat_heat_w=p_bat_heat,
                p_heat_w=max(0.0, therm_ctrl.p_heat_w),
                dt_s=dt_s,
            )
            temp_c = float(therm_state.t_case_c)

            # Keep CPU/battery states aligned with the thermal network
            cpu_state = CpuState(t_j_c=float(therm_state.t_cpu_c), q_thermal_j=cpu_state.q_thermal_j, u_eff=cpu_state.u_eff)
            if battery_soc is not None:
                bat_state = BatterySocState(t_bat_c=float(therm_state.t_bat_c), q_max_ah=bat_state.q_max_ah, soc_avg=bat_state.soc_avg)
                bat_state = battery_aging_step(battery_soc, bat_state, soc_frac=z, t_bat_c=float(therm_state.t_bat_c), dt_s=dt_s)

                q_usable = battery_q_usable_ah(battery_soc, bat_state.q_max_ah, float(therm_state.t_bat_c))
                i_total = max(0.0, i_bat) + max(0.0, battery.i_sd_a)
                dz_dt = -i_total / (q_usable * 3600.0)
                z = clamp01(z + dz_dt * dt_s)
            else:
                # Legacy battery (no advanced model) still uses nominal capacity
                i_total = max(0.0, i_bat) + max(0.0, battery.i_sd_a)
                dz_dt = -i_total / (max(1e-9, battery.c_nom_ah) * 3600.0)
                z = clamp01(z + dz_dt * dt_s)

        t += dt_s

    return t, traces


# -----------------------------
# Traces 字段中文释义/单位表
# -----------------------------

# 说明：
# - simulate_soc() 与 path_planner.plan_optimal_path() 都会返回 traces 字典。
# - 此处提供“键名→中文含义/单位”的统一表，便于后续画图、导出CSV、写报告。


def traces_schema_cn() -> List[Tuple[str, str, str, str]]:
    """返回 traces 字典字段说明表。

    返回值每行是 (key, 中文含义, 单位, 备注)。
    """

    rows: List[Tuple[str, str, str, str]] = [
        ("t_s", "仿真时间", "s", "从 0 开始累计的模拟时间"),
        ("soc", "电池荷电状态SOC（比例）", "1", "范围 0~1"),
        ("soc_pct", "电池荷电状态SOC（百分比）", "%", "=100*soc"),
        (
            "temp_c",
            "手机整体温度代理（与 t_case_c 当前一致）",
            "℃",
            "无三节点热网时为一阶热模型温度；有三节点热网时等于外壳温度",
        ),
        ("t_case_c", "外壳/机身温度", "℃", "当前实现中与 temp_c 同值"),
        ("t_bat_c", "电池温度", "℃", "三节点热网时为电池节点温度；否则来自电池温度代数模型(平滑)"),
        ("q_max_ah", "电池最大可用容量（考虑老化/温度修正）", "Ah", "advanced battery model 开启时会随时间衰减"),
        ("v_oc_v", "电池开路电压（OCV）", "V", "由 SOC→OCV 模型计算"),
        ("i_bat_a", "电池侧放电电流", "A", "用于 SOC 递推的电流；正值表示放电"),
        ("r_bat_ohm", "电池等效内阻", "Ω", "由 SOC/温度模型得到的等效内阻"),
        ("p_device_w", "设备负载功耗（不含电池内部损耗）", "W", "各子系统功耗 + 基线功耗 +（若启用）加热器电耗"),
        ("p_loss_w", "电池内部损耗功率", "W", "内阻损耗 + 放电效率等效损耗"),
        ("p_heat_w", "电池加热器‘输送到电池’的热功率", "W", "热管理控制输出；与 p_heat_elec_w 不同"),
        ("p_heat_elec_w", "加热器从电池消耗的电功率", "W", "= p_heat_w / η_heat(T) 的近似"),
        ("throttle_factor", "热节流因子", "1", "0~1；越小表示越强的性能/需求压缩"),

        ("t_screen_c", "屏幕温度", "℃", "ScreenState.t_s_c"),
        ("t_cpu_c", "CPU结温/核心温度", "℃", "CpuState.t_j_c（若三节点热网启用则与CPU节点对齐）"),

        ("p_sys_w", "系统从电池侧抽取的总功率（SOC更新用）", "W", "通常约等于 p_device_w + p_loss_w"),
        ("p_screen_w", "屏幕功耗", "W", "P_screen = backlight + driver + leakage"),
        ("p_cpu_w", "CPU功耗", "W", "P_cpu = dynamic + static + clock"),
        ("p_wifi_w", "WiFi模块功耗", "W", "含PA/BB/RF/时钟/泄漏等"),
        ("p_bt_w", "蓝牙(BLE)模块功耗", "W", "连接事件平均化模型输出"),
        ("p_gps_w", "GPS模块功耗", "W", "含RF/相关器/DSP/存储/时钟/泄漏等"),
        ("p_spk_w", "扬声器/功放功耗", "W", "Class-D效率曲线 + 音圈焦耳热等效"),
        ("p_cell_w", "蜂窝基带/射频模块功耗", "W", "含PA/BB/RF等"),

        ("l_req", "用户请求亮度（场景输入）", "1", "0~1"),
        ("l_cmd", "控制器输出的亮度命令", "1", "0~1；可能不同于 l_req"),
        ("l_eff", "屏幕有效亮度（考虑响应时间常数）", "1", "0~1；用于功耗与体验评估"),

        ("cpu_demand", "CPU需求负载（场景输入）", "1", "0~1；表示应用对算力的需求"),
        ("cpu_f_ghz", "CPU频率命令", "GHz", "控制器输出（DVFS）"),
        ("cpu_n_active", "CPU激活核心数", "个", "0..n_total"),
        ("cpu_u_eff", "CPU有效利用率", "1", "0~1；考虑动态响应后的实际利用率"),
        ("cpu_u_cmd", "CPU命令利用率", "1", "0~1；控制器输出"),

        ("wifi_r_cmd_bps", "WiFi命令速率", "bps", "控制器输出"),
        ("wifi_r_served_bps", "WiFi实际服务速率", "bps", "受链路容量/状态机影响"),
        ("wifi_q_bits", "WiFi队列长度", "bit", "到达-服务的队列积累"),
        ("wifi_t_c", "WiFi芯片温度", "℃", "热模型输出"),
        ("wifi_p_tx_w", "WiFi发射功率", "W", "功率控制的一阶惯性状态"),
        ("wifi_p_tx_state", "WiFi处于TX状态的概率p5", "1", "WiFiState.p5（CTMC概率）"),

        ("bt_r_cmd_bps", "BLE命令速率", "bps", "控制器输出"),
        ("bt_r_served_bps", "BLE实际服务速率", "bps", "由连接事件占空比与PER等决定"),
        ("bt_q_bits", "BLE队列长度", "bit", "到达-服务的队列积累"),
        ("bt_t_c", "BLE芯片温度", "℃", "热模型输出"),
        ("bt_per", "BLE误包率PER", "1", "0~1；随RSSI平滑变化"),
        ("bt_rssi_dbm", "BLE平均RSSI", "dBm", "RSSI平滑估计"),
        ("bt_p_tx_dbm", "BLE发射功率（dBm）", "dBm", "自适应功率控制状态（离散档位的平滑近似）"),

        ("gps_f_update_hz", "GPS更新率命令", "Hz", "控制器输出；0表示关闭"),
        ("gps_sigma_req_m", "GPS精度要求（越小越严格）", "m", "控制器输出"),
        ("gps_sigma_est_m", "GPS当前精度估计", "m", "模型状态"),
        ("gps_cn0_dbhz", "GPS载噪比C/N0", "dB-Hz", "环境与动态共同决定"),
        ("gps_n_locked", "GPS锁定卫星数（代理）", "颗", "可为连续值（平滑建模）"),
        ("gps_lq", "GPS锁定质量LQ", "1", "0~1"),
        ("gps_t_c", "GPS芯片温度", "℃", "热模型输出"),
        ("gps_m_off", "GPS模式概率：OFF", "1", "0~1"),
        ("gps_m_standby", "GPS模式概率：STANDBY", "1", "0~1"),
        ("gps_m_acq", "GPS模式概率：ACQUISITION", "1", "0~1"),
        ("gps_m_track", "GPS模式概率：TRACKING", "1", "0~1"),
        ("gps_m_nav", "GPS模式概率：NAVIGATION", "1", "0~1"),
        ("gps_m_assist", "GPS模式概率：ASSIST", "1", "0~1"),

        ("spk_g_cmd", "扬声器数字增益命令", "1", "0~1；控制器输出"),
        ("spk_i_vc_a", "音圈电流", "A", "包络/平均化模型状态"),
        ("spk_v_filter_v", "输出滤波电压", "V", "LC输出滤波器等效电压"),
        ("spk_t_vc_c", "音圈温度", "℃", "热模型输出"),
        ("spk_p_joule_w", "音圈焦耳热功率", "W", "= I_vc^2 * R_vc(T)"),
        ("spk_loud_est", "响度估计（代理）", "1", "0~1；由功率映射得到，便于控制"),

        ("cell_r_cmd_bps", "蜂窝命令速率", "bps", "控制器输出"),
        ("cell_r_served_bps", "蜂窝实际服务速率", "bps", "受链路容量/状态机影响"),
        ("cell_q_bits", "蜂窝数据队列长度", "bit", "到达-服务的队列积累"),
        ("cell_t_modem_c", "蜂窝调制解调器温度", "℃", "热模型输出"),
        ("cell_p_tx_w", "蜂窝发射功率", "W", "功率控制状态"),
        ("cell_p_high", "蜂窝处于高活跃状态的概率p3", "1", "CellularState.p3（HIGH_ACTIVITY概率）"),

        # planner 专用（plan_optimal_path 的 traces 中存在）
        ("planner_mode", "规划器选择的离散模式", "-", "eco / balanced / perf；仅在 planner traces 中出现"),
    ]

    return rows


def traces_schema_markdown_cn() -> str:
    """生成 Markdown 格式的 traces 字段说明表（中文）。"""

    rows = traces_schema_cn()
    lines = [
        "# traces 字典字段说明（键名 → 中文含义/单位）",
        "",
        "说明：simulate_soc() 与 path_planner.plan_optimal_path() 返回的 traces 字典字段含义如下。",
        "",
        "| 键名 | 中文含义 | 单位 | 备注 |",
        "|---|---|---|---|",
    ]
    for key, meaning, unit, note in rows:
        # Markdown 表格中尽量避免换行
        safe_note = note.replace("\n", " ")
        lines.append(f"| {key} | {meaning} | {unit} | {safe_note} |")
    lines.append("")
    return "\n".join(lines)
