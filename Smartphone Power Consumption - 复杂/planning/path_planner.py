from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from models.battery_model import (
    BatteryParams,
    BatterySocParams,
    BatterySocState,
    BluetoothControl,
    BluetoothParams,
    BluetoothState,
    CellularControl,
    CellularParams,
    CellularState,
    ComponentPowers,
    CpuControl,
    CpuParams,
    CpuState,
    GpsControl,
    GpsParams,
    GpsState,
    ScreenParams,
    ScreenState,
    SpeakerControl,
    SpeakerParams,
    SpeakerState,
    ThermalParams,
    ThermalControl,
    ThermalNetworkParams,
    ThermalNetworkState,
    UsageInputs,
    WiFiControl,
    WiFiParams,
    WiFiState,
    battery_internal_loss_w,
    battery_t_bat_eq_c,
    battery_ocv_v,
    battery_aging_step,
    battery_q_usable_ah,
    clamp01,
    component_power_w,
    cpu_perf,
    cpu_step,
    cpu_step_workload_only,
    cpu_total_power_w,
    gps_step,
    screen_power_w,
    screen_step,
    speaker_step,
    thermal_step,
    thermal_network_step,
    wifi_step,
    ble_step,
    cellular_step,
    cellular_capacity_bps,
    cellular_expected_power_w,
    voc,
)

from controllers.bluetooth_control import BluetoothCostWeights, BluetoothOneStepOptimalRateController
from controllers.cellular_control import CellularCostWeights, CellularOneStepOptimalRateController
from controllers.cpu_control import CpuCostWeights, CpuOneStepOptimalController
from controllers.gps_control import GpsCostWeights, GpsOneStepOptimalController
from controllers.optimal_control import ScreenCostWeights, ScreenOneStepOptimalController
from controllers.speaker_control import SpeakerCostWeights, SpeakerOneStepOptimalVolumeController
from controllers.wifi_control import WiFiCostWeights, WiFiOneStepOptimalRateController

from controllers.thermal_control import ThermalOneStepController


@dataclass(frozen=True)
class PlannerWeights:
    """Weights for *system-level* rollout evaluation.

    The planner picks a discrete "mode" (ECO/BAL/PERF) by simulating forward
    a short horizon and minimizing the accumulated surrogate cost.
    """

    w_energy: float = 1.0

    # Experience / QoS deficits
    w_screen_track: float = 40.0
    w_cpu_perf: float = 60.0
    w_wifi_def: float = 1.0e-13
    w_bt_def: float = 3.0e-13
    w_cell_def: float = 2.0e-13
    w_gps_update_def: float = 2.0
    w_gps_acc_def: float = 10.0
    w_spk_loud_def: float = 12.0

    # Soft thermal penalty (encourages cooler policies)
    w_temp_soft: float = 0.2
    t_soft_c: float = 45.0

    # Terminal shaping: prefer higher SOC at horizon end (prevents short-sighted PERF)
    w_terminal_soc: float = 8.0


@dataclass(frozen=True)
class ModeSpec:
    name: str

    # Multipliers applied to *controller* weights.
    # Larger power multiplier => more energy saving.
    power_mult: float

    # Larger tracking multiplier => more performance/experience.
    track_mult: float


@dataclass
class SystemState:
    t_s: float
    soc: float
    temp_c: float

    therm_state: Optional[ThermalNetworkState]
    therm_ctrl: ThermalControl

    bat_state: Optional[BatterySocState]

    screen_state: ScreenState
    cpu_state: CpuState
    wifi_state: WiFiState
    bt_state: BluetoothState
    gps_state: GpsState
    spk_state: SpeakerState
    cell_state: CellularState

    # Last applied commands (to warm-start controllers on mode switches)
    l_cmd: float
    cpu_ctrl: CpuControl
    wifi_r_cmd_bps: float
    bt_r_cmd_bps: float
    gps_f_hz: float
    gps_sigma_req_m: float
    spk_g: float
    cell_r_cmd_bps: float


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def default_modes() -> tuple[ModeSpec, ...]:
    return (
        ModeSpec(name="eco", power_mult=2.2, track_mult=0.65),
        ModeSpec(name="balanced", power_mult=1.0, track_mult=1.0),
        ModeSpec(name="perf", power_mult=0.7, track_mult=1.4),
    )


def _make_controllers_for_mode(
    mode: ModeSpec,
    screen: ScreenParams,
    cpu: CpuParams,
    wifi: WiFiParams,
    bt: BluetoothParams,
    gps: Optional[GpsParams],
    spk: Optional[SpeakerParams],
    cell: CellularParams,
    state: SystemState,
) -> dict:
    """Construct a fresh set of controllers for a mode, warm-started from last commands."""

    # Base weights follow simulate.py (the "control" run).
    scr_ctrl = ScreenOneStepOptimalController(
        screen=screen,
        weights=ScreenCostWeights(
            w_track=10.0 * mode.track_mult,
            w_power=1.0 * mode.power_mult,
            w_smooth=0.8,
        ),
        update_interval_s=5.0,
        grid_points=51,
    )
    scr_ctrl.reset(l_cmd0=state.l_cmd)

    cpu_ctrl = CpuOneStepOptimalController(
        cpu=cpu,
        weights=CpuCostWeights(
            w_perf=35.0 * mode.track_mult,
            w_power=0.9 * mode.power_mult,
            w_smooth_f=0.6,
            w_smooth_n=0.2,
            w_smooth_u=0.1,
        ),
        update_interval_s=5.0,
        f_grid=7,
        n_grid=7,
    )
    cpu_ctrl.reset(ctrl0=state.cpu_ctrl)

    wifi_ctrl = WiFiOneStepOptimalRateController(
        params=wifi,
        weights=WiFiCostWeights(
            w_rate=1.0e-13 * mode.track_mult,
            w_power=1.0 * mode.power_mult,
            w_smooth=6.0e-13,
        ),
        update_interval_s=1.0,
    )
    wifi_ctrl.reset(r0_bps=state.wifi_r_cmd_bps)

    bt_ctrl = BluetoothOneStepOptimalRateController(
        params=bt,
        weights=BluetoothCostWeights(
            w_rate=3.0e-13 * mode.track_mult,
            w_power=1.0 * mode.power_mult,
            w_smooth=1.0e-12,
        ),
        update_interval_s=1.0,
        grid_points=21,
    )
    bt_ctrl.reset(r0_bps=state.bt_r_cmd_bps)

    cell_ctrl = CellularOneStepOptimalRateController(
        params=cell,
        weights=CellularCostWeights(
            w_rate=2.0e-13 * mode.track_mult,
            w_power=1.0 * mode.power_mult,
            w_smooth=8.0e-13,
        ),
        update_interval_s=2.0,
        grid_points=21,
    )
    cell_ctrl.reset(r0_bps=state.cell_r_cmd_bps)

    gps_ctrl = None
    if gps is not None:
        gps_ctrl = GpsOneStepOptimalController(
            params=gps,
            weights=GpsCostWeights(
                w_update_deficit=2.0 * mode.track_mult,
                w_acc_deficit=10.0 * mode.track_mult,
                w_power=1.0 * mode.power_mult,
                w_smooth_f=0.20,
                w_smooth_sigma=0.04,
            ),
            update_interval_s=1.0,
        )
        gps_ctrl.reset(f0_hz=state.gps_f_hz, sigma0_m=state.gps_sigma_req_m)

    spk_ctrl = None
    if spk is not None:
        spk_ctrl = SpeakerOneStepOptimalVolumeController(
            params=spk,
            weights=SpeakerCostWeights(
                w_loud_deficit=10.0 * mode.track_mult,
                w_power=1.0 * mode.power_mult,
                w_smooth=0.8,
            ),
            update_interval_s=1.0,
            grid_points=21,
        )
        spk_ctrl.reset(g0=state.spk_g)

    return {
        "screen": scr_ctrl,
        "cpu": cpu_ctrl,
        "wifi": wifi_ctrl,
        "bt": bt_ctrl,
        "gps": gps_ctrl,
        "spk": spk_ctrl,
        "cell": cell_ctrl,
    }


def _required_rate_wifi(wifi: WiFiParams, u: UsageInputs, state: WiFiState) -> float:
    r_arr = max(0.0, u.wifi_rate_bps)
    drain = state.q_wifi_bits / max(1e-3, wifi.target_queue_delay_s)
    return min(wifi.r_max_bps, r_arr + drain)


def _required_rate_bt(bt: BluetoothParams, u: UsageInputs, state: BluetoothState) -> float:
    r_arr = max(0.0, u.bt_arrival_bps)
    drain = state.q_bits / max(1e-3, bt.target_queue_delay_s)
    return min(bt.r_max_bps, r_arr + drain)


def _required_rate_cell(cell: CellularParams, u: UsageInputs, state: CellularState) -> float:
    r_arr = max(0.0, u.cell_arrival_bps)
    drain = state.q_data_bits / max(1e-3, cell.target_queue_delay_s)
    return min(cell.r_max_bps, r_arr + drain)


def _stage_cost(
    weights: PlannerWeights,
    screen: ScreenParams,
    cpu: CpuParams,
    wifi: WiFiParams,
    bt: BluetoothParams,
    cell: CellularParams,
    u_req: UsageInputs,
    u: UsageInputs,
    state: SystemState,
    p_sys_w: float,
    wifi_r_served: float,
    bt_r_served: float,
    cell_r_served: float,
    gps_ctrl: Optional[GpsControl],
    gps_state: GpsState,
    loud_est: float,
    dt_s: float,
) -> float:
    # Energy
    j = weights.w_energy * p_sys_w

    # Screen tracking (use effective brightness)
    l_req = clamp01(u_req.brightness)
    j += weights.w_screen_track * (state.screen_state.l_eff - l_req) ** 2

    # CPU performance deficit
    demand = clamp01(u_req.cpu_demand)
    perf = cpu_perf(cpu, state.cpu_state, state.cpu_ctrl)
    j += weights.w_cpu_perf * (perf - demand) ** 2

    # Network service deficits
    w_req = _required_rate_wifi(wifi, u, state.wifi_state)
    b_req = _required_rate_bt(bt, u, state.bt_state) if clamp01(u.bt_on) >= 0.5 else 0.0
    c_req = _required_rate_cell(cell, u, state.cell_state)

    j += weights.w_wifi_def * (max(0.0, w_req - wifi_r_served) ** 2)
    j += weights.w_bt_def * (max(0.0, b_req - bt_r_served) ** 2)
    j += weights.w_cell_def * (max(0.0, c_req - cell_r_served) ** 2)

    # GPS deficits (relative to app requirements)
    if gps_ctrl is not None and clamp01(u.gps_on) >= 0.5:
        upd_def = max(0.0, u.gps_update_min_hz - gps_ctrl.f_update_hz)
        acc_def = max(0.0, gps_state.sigma_pos_est_m - u.gps_sigma_max_m)
        j += weights.w_gps_update_def * (upd_def**2)
        j += weights.w_gps_acc_def * (acc_def**2)

    # Speaker loudness deficit
    audio_level = _clamp(getattr(u, "spk_audio_level", 0.0), 0.0, 1.0)
    if audio_level > 1e-4:
        loud_req = clamp01(u_req.speaker_volume)
        j += weights.w_spk_loud_def * (max(0.0, loud_req - loud_est) ** 2)

    # Soft thermal penalty
    temp_excess = max(0.0, state.temp_c - weights.t_soft_c)
    j += weights.w_temp_soft * (temp_excess**2)

    return j * dt_s


def _step_system(
    battery: BatteryParams,
    battery_soc: Optional[BatterySocParams],
    thermal: ThermalParams,
    thermal_net: Optional[ThermalNetworkParams],
    powers: ComponentPowers,
    screen: ScreenParams,
    cpu: CpuParams,
    wifi: WiFiParams,
    bt: BluetoothParams,
    gps: Optional[GpsParams],
    spk: Optional[SpeakerParams],
    cell: CellularParams,
    controllers: dict,
    weights: PlannerWeights,
    state: SystemState,
    u_req: UsageInputs,
    t_amb_c: float,
    dt_s: float,
    thermal_controller: Optional[ThermalOneStepController] = None,
) -> tuple[SystemState, dict, float]:
    """Advance the full coupled system by one dt.

    Returns: (next_state, step_metrics, step_cost)
    """

    # Apply last-step thermal throttling to user-level demands (affects other components).
    if thermal_net is not None:
        tf = _clamp(state.therm_ctrl.throttle_factor, 0.0, 1.0)
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

    # Battery voltage proxy for speaker plant/controller.
    v_bat = voc(battery, state.soc) if battery_soc is None else battery_ocv_v(battery_soc, state.soc)

    # Current thermal state (ensure non-None when thermal_net is enabled).
    therm_state = state.therm_state
    if thermal_net is not None and therm_state is None:
        t_bat0 = (state.bat_state.t_bat_c if state.bat_state is not None else state.temp_c)
        therm_state = ThermalNetworkState(t_cpu_c=state.cpu_state.t_j_c, t_bat_c=t_bat0, t_case_c=state.temp_c)

    t_case_c = float(therm_state.t_case_c) if (thermal_net is not None and therm_state is not None) else state.temp_c

    # Screen control -> brightness command
    l_cmd = controllers["screen"](
        state.t_s,
        u_req,
        state.screen_state,
        t_amb_c,
        state.cpu_state.t_j_c,
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

    # CPU control
    cpu_ctrl: CpuControl = controllers["cpu"](state.t_s, u, state.cpu_state, t_amb_c, dt_s)
    if thermal_net is None:
        cpu_state_next = cpu_step(cpu, state.cpu_state, cpu_ctrl, t_amb_c=t_amb_c, dt_s=dt_s)
        cpu_ctrl_eff = cpu_ctrl
        p_cpu = cpu_total_power_w(cpu, cpu_state_next, cpu_ctrl_eff)
    else:
        cpu_state_next, cpu_ctrl_eff, p_cpu = cpu_step_workload_only(cpu, state.cpu_state, cpu_ctrl, dt_s=dt_s)

    # WiFi control
    wifi_ctrl: WiFiControl = controllers["wifi"](
        state.t_s,
        u,
        state.wifi_state,
        t_amb_c,
        (cpu_state_next.t_j_c if thermal_net is None else state.cpu_state.t_j_c),
        state.cell_state.t_modem_c,
        dt_s,
    )
    wifi_state_next, wifi_r_served, p_wifi = wifi_step(
        wifi=wifi,
        state=state.wifi_state,
        ctrl=wifi_ctrl,
        u=u,
        t_amb_c=t_amb_c,
        t_cpu_c=(cpu_state_next.t_j_c if thermal_net is None else state.cpu_state.t_j_c),
        t_cell_c=state.cell_state.t_modem_c,
        dt_s=dt_s,
    )

    # BLE control
    bt_ctrl: BluetoothControl = controllers["bt"](state.t_s, u, state.bt_state, t_amb_c, t_case_c, dt_s)
    bt_state_next, bt_r_served, p_bt = ble_step(
        bt=bt,
        state=state.bt_state,
        ctrl=bt_ctrl,
        u=u,
        t_amb_c=t_amb_c,
        t_adj_c=t_case_c,
        dt_s=dt_s,
    )

    # Cellular control
    t_cpu_for_radios = (cpu_state_next.t_j_c if thermal_net is None else state.cpu_state.t_j_c)
    cell_ctrl: CellularControl = controllers["cell"](state.t_s, u, state.cell_state, t_amb_c, t_cpu_for_radios, dt_s)
    cell_state_next = cellular_step(cell, state.cell_state, cell_ctrl, u, t_amb_c=t_amb_c, t_cpu_c=t_cpu_for_radios, dt_s=dt_s)
    cap = cellular_capacity_bps(cell, cell_state_next)
    cell_r_served = min(cell_ctrl.r_cmd_bps, cap)
    p_cell = cellular_expected_power_w(cell, cell_state_next, cell_ctrl, t_amb_c=t_amb_c, t_cpu_c=t_cpu_for_radios)

    # GPS control
    gps_ctrl_obj = controllers.get("gps")
    gps_ctrl: Optional[GpsControl]
    gps_state_next = state.gps_state
    p_gps = 0.0
    if gps is None or gps_ctrl_obj is None:
        gps_ctrl = None
    else:
        gps_ctrl = gps_ctrl_obj(state.t_s, u, state.gps_state, t_amb_c, t_case_c, dt_s)
        gps_state_next, p_gps = gps_step(
            gps=gps,
            s=state.gps_state,
            c=gps_ctrl,
            u=u,
            t_substrate_c=t_case_c,
            t_shared_c=t_cpu_for_radios,
            dt_s=dt_s,
        )

    # Speaker control
    spk_ctrl_obj = controllers.get("spk")
    spk_ctrl: Optional[SpeakerControl]
    spk_state_next = state.spk_state
    p_spk = 0.0
    p_joule = 0.0
    loud = 0.0
    if spk is None or spk_ctrl_obj is None:
        spk_ctrl = None
    else:
        spk_ctrl = spk_ctrl_obj(state.t_s, u, state.spk_state, t_amb_c, t_case_c, t_cpu_for_radios, v_bat)
        spk_state_next, p_spk, p_joule, loud = speaker_step(
            spk=spk,
            s=state.spk_state,
            c=spk_ctrl,
            u=u,
            t_amb_c=t_amb_c,
            t_phone_c=t_case_c,
            v_bat_v=v_bat,
            dt_s=dt_s,
        )

    # Screen plant
    screen_state_next = screen_step(
        screen=screen,
        s=state.screen_state,
        u=u,
        t_env_c=t_amb_c,
        t_cpu_c=t_cpu_for_radios,
        dt_s=dt_s,
    )
    p_screen = screen_power_w(screen, u, screen_state_next)

    # Other components (lumped)
    other = component_power_w(powers, u, t_case_c)
    p_other = sum(v for k, v in other.items() if k not in ("wifi", "bt", "gps", "speaker"))

    p_device = p_other + p_screen + p_cpu + p_wifi + p_bt + p_gps + p_spk + p_cell

    # Battery internal loss uses device power excluding internal loss.
    if battery_soc is None:
        p_loss, i_bat, v_oc, r_bat = 0.0, p_device / max(1e-6, v_bat), v_bat, 0.0
        p_sys = p_device
    else:
        t_bat_for_r = (state.bat_state.t_bat_c if thermal_net is None else float(therm_state.t_bat_c))
        p_loss, i_bat, v_oc, r_bat = battery_internal_loss_w(battery_soc, state.soc, p_device, t_bat_c=t_bat_for_r)
        p_sys = p_device + p_loss

    # Thermal management (heater + throttling). Heater affects battery drain and battery heat.
    if thermal_net is not None and therm_state is not None and thermal_controller is not None:
        therm_ctrl_next = thermal_controller(
            state.t_s,
            therm_state,
            state.soc,
            t_amb_c,
            p_cpu,
            p_loss,
            dt_s,
        )
    else:
        therm_ctrl_next = ThermalControl(p_heat_w=0.0, p_heat_elec_w=0.0, throttle_factor=1.0)

    if therm_ctrl_next.p_heat_elec_w > 0.0:
        p_device2 = p_device + max(0.0, therm_ctrl_next.p_heat_elec_w)
        if battery_soc is None:
            p_loss2, i_bat2, v_oc2, r_bat2 = 0.0, p_device2 / max(1e-6, v_bat), v_bat, 0.0
            p_sys2 = p_device2
        else:
            t_bat_for_r = (state.bat_state.t_bat_c if thermal_net is None else float(therm_state.t_bat_c))
            p_loss2, i_bat2, v_oc2, r_bat2 = battery_internal_loss_w(battery_soc, state.soc, p_device2, t_bat_c=t_bat_for_r)
            p_sys2 = p_device2 + p_loss2

        p_device, p_loss, i_bat, v_oc, r_bat, p_sys = p_device2, p_loss2, i_bat2, v_oc2, r_bat2, p_sys2

    # Thermal + SOC update
    if thermal_net is None:
        temp_next = thermal_step(thermal, state.temp_c, p_sys, t_amb_c, dt_s)
        therm_state_next = None

        if battery_soc is None:
            i_total = max(0.0, i_bat) + max(0.0, battery.i_sd_a)
            dz_dt = -i_total / (max(1e-9, battery.c_nom_ah) * 3600.0)
            soc_next = clamp01(state.soc + dz_dt * dt_s)
            bat_state_next = None
        else:
            bat_state = state.bat_state or BatterySocState(t_bat_c=state.temp_c, q_max_ah=battery_soc.q_nom_ah, soc_avg=state.soc)
            t_eq = battery_t_bat_eq_c(battery_soc, t_amb_c, p_cpu_w=p_cpu, p_bat_w=p_sys)
            tau_t = max(1e-3, battery_soc.tau_t_bat_s)
            t_bat_next = bat_state.t_bat_c + ((t_eq - bat_state.t_bat_c) / tau_t) * dt_s
            bat_state_mid = BatterySocState(t_bat_c=t_bat_next, q_max_ah=bat_state.q_max_ah, soc_avg=bat_state.soc_avg)
            bat_state_next = battery_aging_step(battery_soc, bat_state_mid, soc_frac=state.soc, t_bat_c=t_bat_next, dt_s=dt_s)

            q_ah = max(1e-9, bat_state_next.q_max_ah)
            i_total = max(0.0, i_bat) + max(0.0, battery.i_sd_a)
            dz_dt = -i_total / (q_ah * 3600.0)
            soc_next = clamp01(state.soc + dz_dt * dt_s)
    else:
        # 3-node thermal network update
        therm_state_next = thermal_network_step(
            thermal_net,
            therm_state,
            t_env_c=t_amb_c,
            p_cpu_w=p_cpu,
            p_bat_heat_w=max(0.0, p_loss),
            p_heat_w=max(0.0, therm_ctrl_next.p_heat_w),
            dt_s=dt_s,
        )
        temp_next = float(therm_state_next.t_case_c)

        # Keep CPU aligned with the thermal network
        cpu_state_next = CpuState(t_j_c=float(therm_state_next.t_cpu_c), q_thermal_j=cpu_state_next.q_thermal_j, u_eff=cpu_state_next.u_eff)

        if battery_soc is None:
            i_total = max(0.0, i_bat) + max(0.0, battery.i_sd_a)
            dz_dt = -i_total / (max(1e-9, battery.c_nom_ah) * 3600.0)
            soc_next = clamp01(state.soc + dz_dt * dt_s)
            bat_state_next = None
        else:
            bat_state = state.bat_state or BatterySocState(t_bat_c=float(therm_state_next.t_bat_c), q_max_ah=battery_soc.q_nom_ah, soc_avg=state.soc)
            bat_state = BatterySocState(t_bat_c=float(therm_state_next.t_bat_c), q_max_ah=bat_state.q_max_ah, soc_avg=bat_state.soc_avg)
            bat_state_next = battery_aging_step(battery_soc, bat_state, soc_frac=state.soc, t_bat_c=float(therm_state_next.t_bat_c), dt_s=dt_s)

            q_usable = battery_q_usable_ah(battery_soc, bat_state_next.q_max_ah, float(therm_state_next.t_bat_c))
            i_total = max(0.0, i_bat) + max(0.0, battery.i_sd_a)
            dz_dt = -i_total / (q_usable * 3600.0)
            soc_next = clamp01(state.soc + dz_dt * dt_s)

    next_state = SystemState(
        t_s=state.t_s + dt_s,
        soc=soc_next,
        temp_c=temp_next,
        therm_state=therm_state_next,
        therm_ctrl=therm_ctrl_next,
        bat_state=bat_state_next,
        screen_state=screen_state_next,
        cpu_state=cpu_state_next,
        wifi_state=wifi_state_next,
        bt_state=bt_state_next,
        gps_state=gps_state_next,
        spk_state=spk_state_next,
        cell_state=cell_state_next,
        l_cmd=clamp01(l_cmd),
        cpu_ctrl=cpu_ctrl_eff,
        wifi_r_cmd_bps=wifi_ctrl.r_cmd_bps,
        bt_r_cmd_bps=bt_ctrl.r_cmd_bps,
        gps_f_hz=(gps_ctrl.f_update_hz if gps_ctrl is not None else 0.0),
        gps_sigma_req_m=(gps_ctrl.sigma_pos_req_m if gps_ctrl is not None else 100.0),
        spk_g=(spk_ctrl.g if spk_ctrl is not None else 0.0),
        cell_r_cmd_bps=cell_ctrl.r_cmd_bps,
    )

    step_cost = _stage_cost(
        weights=weights,
        screen=screen,
        cpu=cpu,
        wifi=wifi,
        bt=bt,
        cell=cell,
        u_req=u_req,
        u=u,
        state=next_state,
        p_sys_w=p_sys,
        wifi_r_served=wifi_r_served,
        bt_r_served=bt_r_served,
        cell_r_served=cell_r_served,
        gps_ctrl=gps_ctrl,
        gps_state=gps_state_next,
        loud_est=loud,
        dt_s=dt_s,
    )

    metrics = {
        "p_sys_w": p_sys,
        "p_device_w": p_device,
        "p_loss_w": p_loss,
        "i_bat_a": i_bat,
        "v_oc_v": v_oc,
        "r_bat_ohm": r_bat,
        "p_heat_w": float(therm_ctrl_next.p_heat_w),
        "p_heat_elec_w": float(therm_ctrl_next.p_heat_elec_w),
        "throttle_factor": float(therm_ctrl_next.throttle_factor),
        "p_screen_w": p_screen,
        "p_cpu_w": p_cpu,
        "p_wifi_w": p_wifi,
        "p_bt_w": p_bt,
        "p_gps_w": p_gps,
        "p_spk_w": p_spk,
        "p_cell_w": p_cell,
        "p_joule_w": p_joule,
        "loud": loud,
        "wifi_r_served": wifi_r_served,
        "bt_r_served": bt_r_served,
        "cell_r_served": cell_r_served,
        "gps_ctrl": gps_ctrl,
    }

    return next_state, metrics, step_cost


def plan_optimal_path(
    battery: BatteryParams,
    battery_soc: Optional[BatterySocParams],
    thermal: ThermalParams,
    thermal_net: Optional[ThermalNetworkParams],
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
    dt_s: float = 2.0,
    rollout_dt_s: float = 10.0,
    t_max_s: float = 24 * 3600,
    decision_interval_s: float = 120.0,
    horizon_s: float = 600.0,
    modes: tuple[ModeSpec, ...] = default_modes(),
    weights: PlannerWeights = PlannerWeights(),
    thermal_controller: Optional[ThermalOneStepController] = None,
) -> Tuple[float, Dict[str, list]]:
    """全程最优路径规划（离散模式 + rollout/多段MPC）。

    每个 decision_interval_s 选择一次模式（eco/balanced/perf），
    通过对每个候选模式做 horizon_s 前向仿真评估总代价，
    选最小者并执行一个决策段，滚动直到 SOC 低于 z_min。

    返回 (t_empty_s, traces)
    """

    # Initialize state similar to simulate_soc baseline.
    z = clamp01(z0)
    bat_state = None
    if battery_soc is not None:
        bat_state = BatterySocState(t_bat_c=t0_c, q_max_ah=battery_soc.q_nom_ah, soc_avg=z)

    f0 = 0.7 * cpu.f_rated_ghz
    n0 = max(1, cpu.n_total // 2)
    cpu_ctrl0 = CpuControl(f_ghz=f0, n_active=n0, u_cmd=0.0)

    state = SystemState(
        t_s=0.0,
        soc=z,
        temp_c=t0_c,
        therm_state=(ThermalNetworkState(t_cpu_c=t0_c, t_bat_c=t0_c, t_case_c=t0_c) if thermal_net is not None else None),
        therm_ctrl=ThermalControl(p_heat_w=0.0, p_heat_elec_w=0.0, throttle_factor=1.0),
        bat_state=bat_state,
        screen_state=ScreenState(t_s_c=t0_c, q_pixel=0.0, l_eff=0.0),
        cpu_state=CpuState(t_j_c=t0_c, q_thermal_j=0.0, u_eff=0.0),
        wifi_state=WiFiState(t_wifi_c=t0_c),
        bt_state=BluetoothState(t_ble_c=t0_c),
        gps_state=GpsState(t_gps_c=t0_c),
        spk_state=SpeakerState(t_vc_c=t0_c),
        cell_state=CellularState(t_modem_c=t0_c),
        l_cmd=0.2,
        cpu_ctrl=cpu_ctrl0,
        wifi_r_cmd_bps=0.0,
        bt_r_cmd_bps=0.0,
        gps_f_hz=0.0,
        gps_sigma_req_m=50.0,
        spk_g=0.0,
        cell_r_cmd_bps=0.0,
    )

    traces: Dict[str, list] = {
        "t_s": [],
        "soc": [],
        "temp_c": [],
        "t_case_c": [],
        "t_cpu_c": [],
        "t_bat_c": [],
        "q_max_ah": [],
        "p_heat_w": [],
        "p_heat_elec_w": [],
        "throttle_factor": [],
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
        "planner_mode": [],

        # battery-level diagnostics (optional)
        "p_device_w": [],
        "p_loss_w": [],
        "i_bat_a": [],
        "v_oc_v": [],
        "r_bat_ohm": [],
    }

    decision_steps = max(1, int(round(decision_interval_s / dt_s)))
    dt_eval = max(dt_s, float(rollout_dt_s))
    horizon_steps = max(1, int(round(horizon_s / dt_eval)))

    while state.t_s < t_max_s and state.soc > z_min:
        # Pick the best mode by rollout
        best_mode = modes[0]
        best_j = float("inf")

        u_req0 = usage_fn(state.t_s)

        for mode in modes:
            # Fresh controllers for candidate mode, warm-started
            controllers = _make_controllers_for_mode(mode, screen, cpu, wifi, bluetooth, gps, speaker, cellular, state)

            s_sim: SystemState = copy.deepcopy(state)
            j_total = 0.0

            for k in range(horizon_steps):
                if s_sim.soc <= z_min or (s_sim.t_s >= t_max_s):
                    break
                u_req = usage_fn(s_sim.t_s)
                s_sim, _metrics, j_step = _step_system(
                    battery=battery,
                    battery_soc=battery_soc,
                    thermal=thermal,
                    thermal_net=thermal_net,
                    powers=powers,
                    screen=screen,
                    cpu=cpu,
                    wifi=wifi,
                    bt=bluetooth,
                    gps=gps,
                    spk=speaker,
                    cell=cellular,
                    controllers=controllers,
                    weights=weights,
                    state=s_sim,
                    u_req=u_req,
                    t_amb_c=t_amb_c,
                    dt_s=dt_eval,
                    thermal_controller=thermal_controller,
                )
                j_total += j_step

                # Early pruning
                if j_total >= best_j:
                    break

            # terminal shaping
            j_total += weights.w_terminal_soc * (1.0 - s_sim.soc)

            if j_total < best_j:
                best_j = j_total
                best_mode = mode

        # Execute one decision segment with the best mode
        controllers = _make_controllers_for_mode(best_mode, screen, cpu, wifi, bluetooth, gps, speaker, cellular, state)

        for _ in range(decision_steps):
            if state.soc <= z_min or (state.t_s >= t_max_s):
                break
            u_req = usage_fn(state.t_s)
            state, metrics, _j_step = _step_system(
                battery=battery,
                battery_soc=battery_soc,
                thermal=thermal,
                thermal_net=thermal_net,
                powers=powers,
                screen=screen,
                cpu=cpu,
                wifi=wifi,
                bt=bluetooth,
                gps=gps,
                spk=speaker,
                cell=cellular,
                controllers=controllers,
                weights=weights,
                state=state,
                u_req=u_req,
                t_amb_c=t_amb_c,
                dt_s=dt_s,
                thermal_controller=thermal_controller,
            )

            # Record traces in the same schema as simulate_soc for easy comparison
            traces["t_s"].append(state.t_s)
            traces["soc"].append(state.soc)
            traces["temp_c"].append(state.temp_c)
            traces["t_case_c"].append(state.temp_c)
            traces["t_cpu_c"].append(float(state.cpu_state.t_j_c))
            if state.bat_state is None:
                traces["t_bat_c"].append(float(state.temp_c))
                traces["q_max_ah"].append(float(battery.c_nom_ah))
            else:
                traces["t_bat_c"].append(float(state.bat_state.t_bat_c))
                traces["q_max_ah"].append(float(state.bat_state.q_max_ah))

            traces["p_heat_w"].append(float(state.therm_ctrl.p_heat_w))
            traces["p_heat_elec_w"].append(float(state.therm_ctrl.p_heat_elec_w))
            traces["throttle_factor"].append(float(state.therm_ctrl.throttle_factor))
            traces["p_sys_w"].append(metrics["p_sys_w"])
            traces["p_device_w"].append(metrics["p_device_w"])
            traces["p_loss_w"].append(metrics["p_loss_w"])
            traces["i_bat_a"].append(metrics["i_bat_a"])
            traces["v_oc_v"].append(metrics["v_oc_v"])
            traces["r_bat_ohm"].append(metrics["r_bat_ohm"])

            traces["p_screen_w"].append(metrics["p_screen_w"])
            traces["p_cpu_w"].append(metrics["p_cpu_w"])
            traces["p_wifi_w"].append(metrics["p_wifi_w"])
            traces["p_bt_w"].append(metrics["p_bt_w"])
            traces["p_gps_w"].append(metrics["p_gps_w"])
            traces["p_spk_w"].append(metrics["p_spk_w"])
            traces["p_cell_w"].append(metrics["p_cell_w"])

            traces["l_req"].append(clamp01(u_req.brightness))
            traces["l_cmd"].append(clamp01(state.l_cmd))
            traces["l_eff"].append(clamp01(state.screen_state.l_eff))

            traces["cpu_demand"].append(clamp01(u_req.cpu_demand))
            traces["cpu_f_ghz"].append(float(state.cpu_ctrl.f_ghz))
            traces["cpu_n_active"].append(int(state.cpu_ctrl.n_active))
            traces["cpu_u_eff"].append(clamp01(state.cpu_state.u_eff))
            traces["cpu_u_cmd"].append(clamp01(state.cpu_ctrl.u_cmd))

            traces["wifi_r_cmd_bps"].append(float(state.wifi_r_cmd_bps))
            traces["wifi_r_served_bps"].append(float(metrics["wifi_r_served"]))
            traces["wifi_q_bits"].append(float(state.wifi_state.q_wifi_bits))
            traces["wifi_t_c"].append(float(state.wifi_state.t_wifi_c))
            traces["wifi_p_tx_w"].append(float(state.wifi_state.p_tx_w))
            traces["wifi_p_tx_state"].append(float(state.wifi_state.p5))

            traces["bt_r_cmd_bps"].append(float(state.bt_r_cmd_bps))
            traces["bt_r_served_bps"].append(float(metrics["bt_r_served"]))
            traces["bt_q_bits"].append(float(state.bt_state.q_bits))
            traces["bt_t_c"].append(float(state.bt_state.t_ble_c))
            traces["bt_per"].append(float(state.bt_state.per))
            traces["bt_rssi_dbm"].append(float(state.bt_state.rssi_avg_dbm))
            traces["bt_p_tx_dbm"].append(float(state.bt_state.p_tx_dbm))

            gps_ctrl: Optional[GpsControl] = metrics["gps_ctrl"]
            traces["gps_f_update_hz"].append(float(gps_ctrl.f_update_hz) if gps_ctrl is not None else 0.0)
            traces["gps_sigma_req_m"].append(float(gps_ctrl.sigma_pos_req_m) if gps_ctrl is not None else 100.0)
            traces["gps_sigma_est_m"].append(float(state.gps_state.sigma_pos_est_m))
            traces["gps_cn0_dbhz"].append(float(state.gps_state.cn0_dbhz))
            traces["gps_n_locked"].append(float(state.gps_state.n_locked))
            traces["gps_lq"].append(float(state.gps_state.lq))
            traces["gps_t_c"].append(float(state.gps_state.t_gps_c))
            traces["gps_m_off"].append(float(state.gps_state.m0))
            traces["gps_m_standby"].append(float(state.gps_state.m1))
            traces["gps_m_acq"].append(float(state.gps_state.m2))
            traces["gps_m_track"].append(float(state.gps_state.m3))
            traces["gps_m_nav"].append(float(state.gps_state.m4))
            traces["gps_m_assist"].append(float(state.gps_state.m5))

            traces["spk_g_cmd"].append(float(state.spk_g))
            traces["spk_i_vc_a"].append(float(state.spk_state.i_vc_a))
            traces["spk_v_filter_v"].append(float(state.spk_state.v_filter_v))
            traces["spk_t_vc_c"].append(float(state.spk_state.t_vc_c))
            traces["spk_p_joule_w"].append(float(metrics["p_joule_w"]))
            traces["spk_loud_est"].append(float(metrics["loud"]))

            traces["cell_r_cmd_bps"].append(float(state.cell_r_cmd_bps))
            traces["cell_r_served_bps"].append(float(metrics["cell_r_served"]))
            traces["cell_q_bits"].append(float(state.cell_state.q_data_bits))
            traces["cell_t_modem_c"].append(float(state.cell_state.t_modem_c))
            traces["cell_p_tx_w"].append(float(state.cell_state.p_tx_w))
            traces["cell_p_high"].append(float(state.cell_state.p3))

            traces["planner_mode"].append(best_mode.name)

    return state.t_s, traces
