from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


# 允许直接运行：python sensitivity_analysis/run_sensitivity.py
# 此时 sys.path[0] 是 sensitivity_analysis/，需要手动把工程根目录加入搜索路径。
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.battery_model import (
    BatteryParams,
    BatterySocParams,
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
    ThermalNetworkParams,
    ThermalNetworkState,
    ThermalParams,
    ThermalControl,
    UsageInputs,
    WiFiControl,
    WiFiParams,
    WiFiState,
    clamp01,
    simulate_soc,
)

from controllers.bluetooth_control import BluetoothCostWeights, BluetoothOneStepOptimalRateController
from controllers.cellular_control import CellularCostWeights, CellularOneStepOptimalRateController
from controllers.cpu_control import CpuCostWeights, CpuOneStepOptimalController
from controllers.gps_control import GpsCostWeights, GpsOneStepOptimalController
from controllers.optimal_control import ScreenCostWeights, ScreenOneStepOptimalController
from controllers.speaker_control import SpeakerCostWeights, SpeakerOneStepOptimalVolumeController
from controllers.thermal_control import ThermalOneStepController
from controllers.wifi_control import WiFiCostWeights, WiFiOneStepOptimalRateController

from scenario_lib import scenarios


@dataclass(frozen=True)
class SensitivityResult:
    key: str
    cn_name: str
    unit: str
    baseline_u_mean: float
    delta_u: float
    t0_s: float
    t1_s: float
    sensitivity_s_per_unit: float
    elasticity: float


def _mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-12 else float("nan")


def _wrap_screen(ctrl: Callable, eps: float) -> Callable:
    def wrapped(t_s: float, u_req: UsageInputs, screen_state: ScreenState, t_env_c: float, t_cpu_c: float, dt_s: float) -> float:
        l = float(ctrl(t_s, u_req, screen_state, t_env_c, t_cpu_c, dt_s))
        return clamp01(l * (1.0 + eps))

    return wrapped


def _wrap_cpu_freq(ctrl: Callable, cpu: CpuParams, eps: float) -> Callable:
    def wrapped(t_s: float, u_req: UsageInputs, state: CpuState, t_amb_c: float, dt_s: float) -> CpuControl:
        c: CpuControl = ctrl(t_s, u_req, state, t_amb_c, dt_s)
        f = max(cpu.f_min_ghz, min(cpu.f_rated_ghz, float(c.f_ghz) * (1.0 + eps)))
        return CpuControl(f_ghz=f, n_active=int(c.n_active), u_cmd=clamp01(float(c.u_cmd)))

    return wrapped


def _wrap_cpu_u(ctrl: Callable, cpu: CpuParams, eps: float) -> Callable:
    def wrapped(t_s: float, u_req: UsageInputs, state: CpuState, t_amb_c: float, dt_s: float) -> CpuControl:
        c: CpuControl = ctrl(t_s, u_req, state, t_amb_c, dt_s)
        return CpuControl(f_ghz=float(c.f_ghz), n_active=int(c.n_active), u_cmd=clamp01(float(c.u_cmd) * (1.0 + eps)))

    return wrapped


def _wrap_cpu_n(ctrl: Callable, cpu: CpuParams, delta_n: int) -> Callable:
    def wrapped(t_s: float, u_req: UsageInputs, state: CpuState, t_amb_c: float, dt_s: float) -> CpuControl:
        c: CpuControl = ctrl(t_s, u_req, state, t_amb_c, dt_s)
        n = max(0, min(cpu.n_total, int(c.n_active) + int(delta_n)))
        return CpuControl(f_ghz=float(c.f_ghz), n_active=int(n), u_cmd=clamp01(float(c.u_cmd)))

    return wrapped


def _wrap_rate_bps(ctrl: Callable, r_max_bps: float, eps: float, ctor) -> Callable:
    def wrapped(*args, **kwargs):
        c = ctrl(*args, **kwargs)
        r = max(0.0, min(r_max_bps, float(c.r_cmd_bps) * (1.0 + eps)))
        return ctor(r_cmd_bps=r)

    return wrapped


def _wrap_gps_f(ctrl: Callable, eps: float) -> Callable:
    def wrapped(t_s: float, u: UsageInputs, state: GpsState, t_amb_c: float, t_substrate_c: float, dt_s: float) -> GpsControl:
        c: GpsControl = ctrl(t_s, u, state, t_amb_c, t_substrate_c, dt_s)
        f = max(0.0, min(10.0, float(c.f_update_hz) * (1.0 + eps)))
        return GpsControl(
            sigma_pos_req_m=float(c.sigma_pos_req_m),
            f_update_hz=f,
            cn0_thresh_dbhz=float(c.cn0_thresh_dbhz),
            b_loop_hz=float(c.b_loop_hz),
            assist=float(c.assist),
        )

    return wrapped


def _wrap_gps_sigma(ctrl: Callable, eps: float) -> Callable:
    def wrapped(t_s: float, u: UsageInputs, state: GpsState, t_amb_c: float, t_substrate_c: float, dt_s: float) -> GpsControl:
        c: GpsControl = ctrl(t_s, u, state, t_amb_c, t_substrate_c, dt_s)
        sigma = max(1.0, min(100.0, float(c.sigma_pos_req_m) * (1.0 + eps)))
        return GpsControl(
            sigma_pos_req_m=sigma,
            f_update_hz=float(c.f_update_hz),
            cn0_thresh_dbhz=float(c.cn0_thresh_dbhz),
            b_loop_hz=float(c.b_loop_hz),
            assist=float(c.assist),
        )

    return wrapped


def _wrap_speaker_gain(ctrl: Callable, eps: float) -> Callable:
    def wrapped(t_s: float, u: UsageInputs, state: SpeakerState, t_amb_c: float, t_phone_c: float, t_cpu_c: float, v_bat_v: float) -> SpeakerControl:
        c: SpeakerControl = ctrl(t_s, u, state, t_amb_c, t_phone_c, t_cpu_c, v_bat_v)
        g = clamp01(float(c.g) * (1.0 + eps))
        return SpeakerControl(g=g, v_limit_v=float(c.v_limit_v), f_mode=int(c.f_mode))

    return wrapped


def _wrap_heater(ctrl: Callable, eps: float) -> Callable:
    def wrapped(
        t_s: float,
        therm: ThermalNetworkState,
        soc: float,
        t_env_c: float,
        p_cpu_w: float,
        p_bat_heat_wo_heater_w: float,
        dt_s: float,
    ) -> ThermalControl:
        c: ThermalControl = ctrl(t_s, therm, soc, t_env_c, p_cpu_w, p_bat_heat_wo_heater_w, dt_s)
        p_heat = max(0.0, float(c.p_heat_w) * (1.0 + eps))
        p_elec = max(0.0, float(c.p_heat_elec_w) * (1.0 + eps))
        return ThermalControl(p_heat_w=p_heat, p_heat_elec_w=p_elec, throttle_factor=float(c.throttle_factor))

    return wrapped


def _get_scenario(name: str):
    if not hasattr(scenarios, name):
        raise SystemExit(f"未知场景: {name}. 可用: airplane_mode_reading, piecewise_day_scenario, worst_case_nav_hotspot")
    fn = getattr(scenarios, name)
    return fn()


def _build_controllers(
    screen: ScreenParams,
    cpu: CpuParams,
    wifi: WiFiParams,
    bt: BluetoothParams,
    gps: Optional[GpsParams],
    spk: Optional[SpeakerParams],
    cell: CellularParams,
) -> Dict[str, object]:
    thermal_controller = ThermalOneStepController(update_interval_s=5.0)

    scr_ctrl = ScreenOneStepOptimalController(
        screen=screen,
        weights=ScreenCostWeights(w_track=10.0, w_power=1.0, w_smooth=0.8),
        update_interval_s=5.0,
        grid_points=51,
    )

    cpu_ctrl = CpuOneStepOptimalController(
        cpu=cpu,
        weights=CpuCostWeights(w_perf=35.0, w_power=0.9, w_smooth_f=0.6, w_smooth_n=0.2, w_smooth_u=0.1),
        update_interval_s=5.0,
        f_grid=7,
        n_grid=7,
    )

    wifi_ctrl = WiFiOneStepOptimalRateController(
        params=wifi,
        weights=WiFiCostWeights(w_rate=1.0e-13, w_power=1.0, w_smooth=6.0e-13),
        update_interval_s=1.0,
    )

    bt_ctrl = BluetoothOneStepOptimalRateController(
        params=bt,
        weights=BluetoothCostWeights(w_rate=3.0e-13, w_power=1.0, w_smooth=1.0e-12),
        update_interval_s=1.0,
        grid_points=21,
    )

    cell_ctrl = CellularOneStepOptimalRateController(
        params=cell,
        weights=CellularCostWeights(w_rate=2.0e-13, w_power=1.0, w_smooth=8.0e-13),
        update_interval_s=2.0,
        grid_points=21,
    )

    gps_ctrl = None
    if gps is not None:
        gps_ctrl = GpsOneStepOptimalController(
            params=gps,
            weights=GpsCostWeights(w_update_deficit=2.0, w_acc_deficit=10.0, w_power=1.0, w_smooth_f=0.20, w_smooth_sigma=0.04),
            update_interval_s=1.0,
        )

    spk_ctrl = None
    if spk is not None:
        spk_ctrl = SpeakerOneStepOptimalVolumeController(
            params=spk,
            weights=SpeakerCostWeights(w_loud_deficit=10.0, w_power=1.0, w_smooth=0.8),
            update_interval_s=1.0,
            grid_points=21,
        )

    return {
        "thermal": thermal_controller,
        "screen": scr_ctrl,
        "cpu": cpu_ctrl,
        "wifi": wifi_ctrl,
        "bt": bt_ctrl,
        "cell": cell_ctrl,
        "gps": gps_ctrl,
        "spk": spk_ctrl,
    }


def _run_once(
    usage_fn: Callable[[float], UsageInputs],
    t_amb_c: float,
    z0: float,
    z_min: float,
    dt_s: float,
    t_max_s: float,
    controllers: Dict[str, object],
) -> Tuple[float, Dict[str, list]]:
    battery = BatteryParams(c_nom_ah=4.5, aging_alpha=0.10, eta_pmu=0.90, i_sd_a=0.0)
    battery_soc = BatterySocParams(q_nom_ah=battery.c_nom_ah, v_nom_v=0.5 * (battery.v_min + battery.v_max))

    thermal = ThermalParams(c_th_j_per_c=40.0, r_th_c_per_w=7.0)
    thermal_net = ThermalNetworkParams()

    powers = ComponentPowers()
    screen = ScreenParams()
    cpu = CpuParams()
    wifi = WiFiParams()
    bt = BluetoothParams()
    gps = GpsParams()
    spk = SpeakerParams()
    cellular = CellularParams()

    # Controllers are provided from caller, but they were created with different params.
    # For sensitivity analysis we must rebuild controllers with the *same* params objects.
    ctrls = _build_controllers(screen, cpu, wifi, bt, gps, spk, cellular)

    # Optionally override with wrapped controllers.
    ctrls.update(controllers)

    return simulate_soc(
        battery=battery,
        battery_soc=battery_soc,
        thermal=thermal,
        thermal_net=thermal_net,
        thermal_controller=ctrls["thermal"],
        powers=powers,
        screen=screen,
        cpu=cpu,
        wifi=wifi,
        bluetooth=bt,
        gps=gps,
        speaker=spk,
        cellular=cellular,
        usage_fn=usage_fn,
        z0=z0,
        z_min=z_min,
        t_amb_c=t_amb_c,
        t0_c=t_amb_c,
        dt_s=dt_s,
        t_max_s=t_max_s,
        screen_brightness_controller=ctrls["screen"],
        cpu_controller=ctrls["cpu"],
        wifi_controller=ctrls["wifi"],
        bluetooth_controller=ctrls["bt"],
        gps_controller=ctrls["gps"],
        speaker_controller=ctrls["spk"],
        cellular_controller=ctrls["cell"],
    )


def analyze_sensitivities(
    scenario_name: str,
    t_amb_c: float,
    eps: float,
    z0: float,
    z_min: float,
    dt_s: float,
    t_max_s: float,
) -> list[SensitivityResult]:
    sc = _get_scenario(scenario_name)

    # Baseline
    t0_s, tr0 = _run_once(
        usage_fn=sc.usage_fn,
        t_amb_c=t_amb_c,
        z0=z0,
        z_min=z_min,
        dt_s=dt_s,
        t_max_s=t_max_s,
        controllers={},
    )

    results: list[SensitivityResult] = []

    def add_result(key: str, cn: str, unit: str, u_key: str, t1_s: float, delta_u: float) -> None:
        u_bar = float(_mean([float(x) for x in tr0.get(u_key, [])]))
        sens = _safe_div((t1_s - t0_s), delta_u)
        elas = _safe_div((t1_s - t0_s) / max(1e-12, t0_s), delta_u / max(1e-12, u_bar))
        results.append(
            SensitivityResult(
                key=key,
                cn_name=cn,
                unit=unit,
                baseline_u_mean=u_bar,
                delta_u=delta_u,
                t0_s=float(t0_s),
                t1_s=float(t1_s),
                sensitivity_s_per_unit=float(sens),
                elasticity=float(elas),
            )
        )

    # We need param objects only for clamping in wrappers; use fresh default params.
    cpu = CpuParams()
    wifi = WiFiParams()
    bt = BluetoothParams()
    cell = CellularParams()

    # 1) Screen brightness command l_cmd
    t1_s, _ = _run_once(sc.usage_fn, t_amb_c, z0, z_min, dt_s, t_max_s, controllers={"screen": _wrap_screen(_build_controllers(ScreenParams(), cpu, wifi, bt, GpsParams(), SpeakerParams(), cell)["screen"], eps)})
    add_result("l_cmd", "屏幕亮度命令", "1", "l_cmd", t1_s, delta_u=max(1e-12, _mean(tr0.get("l_cmd", [0.0])) * eps))

    # 2) CPU controls
    base_cpu_ctrl = _build_controllers(ScreenParams(), cpu, wifi, bt, GpsParams(), SpeakerParams(), cell)["cpu"]

    t1_s, _ = _run_once(sc.usage_fn, t_amb_c, z0, z_min, dt_s, t_max_s, controllers={"cpu": _wrap_cpu_freq(base_cpu_ctrl, cpu, eps)})
    add_result("cpu_f_ghz", "CPU频率命令", "GHz", "cpu_f_ghz", t1_s, delta_u=max(1e-12, _mean(tr0.get("cpu_f_ghz", [0.0])) * eps))

    t1_s, _ = _run_once(sc.usage_fn, t_amb_c, z0, z_min, dt_s, t_max_s, controllers={"cpu": _wrap_cpu_u(base_cpu_ctrl, cpu, eps)})
    add_result("cpu_u_cmd", "CPU利用率命令", "1", "cpu_u_cmd", t1_s, delta_u=max(1e-12, _mean(tr0.get("cpu_u_cmd", [0.0])) * eps))

    # 核数是整数：用 +1 核的有限差分
    t1_s, _ = _run_once(sc.usage_fn, t_amb_c, z0, z_min, dt_s, t_max_s, controllers={"cpu": _wrap_cpu_n(base_cpu_ctrl, cpu, delta_n=1)})
    add_result("cpu_n_active", "CPU激活核心数", "个", "cpu_n_active", t1_s, delta_u=1.0)

    # 3) Network rates
    base_wifi_ctrl = _build_controllers(ScreenParams(), cpu, wifi, bt, GpsParams(), SpeakerParams(), cell)["wifi"]
    t1_s, _ = _run_once(sc.usage_fn, t_amb_c, z0, z_min, dt_s, t_max_s, controllers={"wifi": _wrap_rate_bps(base_wifi_ctrl, wifi.r_max_bps, eps, WiFiControl)})
    add_result("wifi_r_cmd_bps", "WiFi命令速率", "bps", "wifi_r_cmd_bps", t1_s, delta_u=max(1e-12, _mean(tr0.get("wifi_r_cmd_bps", [0.0])) * eps))

    base_bt_ctrl = _build_controllers(ScreenParams(), cpu, wifi, bt, GpsParams(), SpeakerParams(), cell)["bt"]
    t1_s, _ = _run_once(sc.usage_fn, t_amb_c, z0, z_min, dt_s, t_max_s, controllers={"bt": _wrap_rate_bps(base_bt_ctrl, bt.r_max_bps, eps, BluetoothControl)})
    add_result("bt_r_cmd_bps", "BLE命令速率", "bps", "bt_r_cmd_bps", t1_s, delta_u=max(1e-12, _mean(tr0.get("bt_r_cmd_bps", [0.0])) * eps))

    base_cell_ctrl = _build_controllers(ScreenParams(), cpu, wifi, bt, GpsParams(), SpeakerParams(), cell)["cell"]
    t1_s, _ = _run_once(sc.usage_fn, t_amb_c, z0, z_min, dt_s, t_max_s, controllers={"cell": _wrap_rate_bps(base_cell_ctrl, cell.r_max_bps, eps, CellularControl)})
    add_result("cell_r_cmd_bps", "蜂窝命令速率", "bps", "cell_r_cmd_bps", t1_s, delta_u=max(1e-12, _mean(tr0.get("cell_r_cmd_bps", [0.0])) * eps))

    # 4) GPS
    base_gps_ctrl = _build_controllers(ScreenParams(), cpu, wifi, bt, GpsParams(), SpeakerParams(), cell)["gps"]
    if base_gps_ctrl is not None:
        t1_s, _ = _run_once(sc.usage_fn, t_amb_c, z0, z_min, dt_s, t_max_s, controllers={"gps": _wrap_gps_f(base_gps_ctrl, eps)})
        add_result("gps_f_update_hz", "GPS更新率命令", "Hz", "gps_f_update_hz", t1_s, delta_u=max(1e-12, _mean(tr0.get("gps_f_update_hz", [0.0])) * eps))

        t1_s, _ = _run_once(sc.usage_fn, t_amb_c, z0, z_min, dt_s, t_max_s, controllers={"gps": _wrap_gps_sigma(base_gps_ctrl, eps)})
        add_result("gps_sigma_req_m", "GPS精度要求命令", "m", "gps_sigma_req_m", t1_s, delta_u=max(1e-12, _mean(tr0.get("gps_sigma_req_m", [0.0])) * eps))

    # 5) Speaker
    base_spk_ctrl = _build_controllers(ScreenParams(), cpu, wifi, bt, GpsParams(), SpeakerParams(), cell)["spk"]
    if base_spk_ctrl is not None:
        t1_s, _ = _run_once(sc.usage_fn, t_amb_c, z0, z_min, dt_s, t_max_s, controllers={"spk": _wrap_speaker_gain(base_spk_ctrl, eps)})
        add_result("spk_g_cmd", "扬声器增益命令", "1", "spk_g_cmd", t1_s, delta_u=max(1e-12, _mean(tr0.get("spk_g_cmd", [0.0])) * eps))

    # 6) Heater
    base_therm_ctrl = _build_controllers(ScreenParams(), cpu, wifi, bt, GpsParams(), SpeakerParams(), cell)["thermal"]
    t1_s, _ = _run_once(sc.usage_fn, t_amb_c, z0, z_min, dt_s, t_max_s, controllers={"thermal": _wrap_heater(base_therm_ctrl, eps)})
    add_result("p_heat_w", "加热器热功率命令", "W", "p_heat_w", t1_s, delta_u=max(1e-12, _mean(tr0.get("p_heat_w", [0.0])) * eps))

    return results


def _print_table(rows: list[SensitivityResult]) -> None:
    print("\n=== 被控量对 t_empty 的局部灵敏度（数值扰动法） ===")
    print("字段 | 中文 | ū | Δu | t0(s) | t1(s) | S=(Δt/Δu) | 弹性E")
    for r in rows:
        print(
            f"{r.key:14s} | {r.cn_name:10s} | {r.baseline_u_mean:9.3g} | {r.delta_u:9.3g} | "
            f"{r.t0_s:9.3g} | {r.t1_s:9.3g} | {r.sensitivity_s_per_unit:11.3g} | {r.elasticity:7.3g}"
        )


def _write_csv(path: str, rows: list[SensitivityResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "key",
                "cn_name",
                "unit",
                "baseline_u_mean",
                "delta_u",
                "t0_s",
                "t1_s",
                "sensitivity_s_per_unit",
                "elasticity",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.key,
                    r.cn_name,
                    r.unit,
                    r.baseline_u_mean,
                    r.delta_u,
                    r.t0_s,
                    r.t1_s,
                    r.sensitivity_s_per_unit,
                    r.elasticity,
                ]
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="worst_case_nav_hotspot")
    ap.add_argument("--t-amb", type=float, default=0.0)
    ap.add_argument("--eps", type=float, default=0.01, help="相对扰动幅度，例如 0.01=+1%%")
    ap.add_argument("--z0", type=float, default=0.30)
    ap.add_argument("--z-min", type=float, default=0.28)
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--t-max-s", type=float, default=30 * 60)
    ap.add_argument("--out", default=None, help="输出 CSV 路径")
    args = ap.parse_args()

    rows = analyze_sensitivities(
        scenario_name=args.scenario,
        t_amb_c=float(args.t_amb),
        eps=float(args.eps),
        z0=float(args.z0),
        z_min=float(args.z_min),
        dt_s=float(args.dt),
        t_max_s=float(args.t_max_s),
    )

    _print_table(rows)
    if args.out:
        _write_csv(args.out, rows)
        print(f"\n已写出 CSV：{args.out}")


if __name__ == "__main__":
    main()
