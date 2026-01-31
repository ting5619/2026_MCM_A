from __future__ import annotations

import math
import os
import sys


# 允许直接运行：python scripts/simulate.py
# 此时 sys.path[0] 是 scripts/，需要手动把工程根目录加入搜索路径。
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.battery_model import (
    BatteryParams,
    BatterySocParams,
    BluetoothParams,
    CellularParams,
    ComponentPowers,
    CpuParams,
    GpsParams,
    ScreenParams,
    SpeakerParams,
    ThermalParams,
    ThermalNetworkParams,
    WiFiParams,
    simulate_soc,
    traces_schema_markdown_cn,
)
from controllers.bluetooth_control import BluetoothCostWeights, BluetoothOneStepOptimalRateController
from controllers.cellular_control import CellularCostWeights, CellularOneStepOptimalRateController
from controllers.cpu_control import CpuCostWeights, CpuOneStepOptimalController
from controllers.gps_control import GpsCostWeights, GpsOneStepOptimalController
from controllers.optimal_control import ScreenCostWeights, ScreenOneStepOptimalController
from scenario_lib.scenarios import airplane_mode_reading, piecewise_day_scenario, worst_case_nav_hotspot
from controllers.speaker_control import SpeakerCostWeights, SpeakerOneStepOptimalVolumeController
from controllers.wifi_control import WiFiCostWeights, WiFiOneStepOptimalRateController

from planning.path_planner import plan_optimal_path
from controllers.thermal_control import ThermalOneStepController


def fmt_hours(seconds: float) -> str:
    h = seconds / 3600.0
    if h < 1:
        return f"{h*60:.1f} min"
    return f"{h:.2f} h"


def run_one(name: str, z0: float, t_amb_c: float) -> None:
    battery = BatteryParams(
        c_nom_ah=4.5,
        aging_alpha=0.10,  # 假设已老化10%
        eta_pmu=0.90,
        i_sd_a=0.0,
    )

    # Advanced battery internal-loss/OCV/aging model (can be set to None to use the legacy model)
    battery_soc = BatterySocParams(
        q_nom_ah=battery.c_nom_ah,
        v_nom_v=0.5 * (battery.v_min + battery.v_max),
    )

    thermal = ThermalParams(c_th_j_per_c=40.0, r_th_c_per_w=7.0)
    thermal_net = ThermalNetworkParams()
    thermal_controller = ThermalOneStepController(update_interval_s=5.0)
    powers = ComponentPowers()
    screen = ScreenParams()
    cpu = CpuParams()
    wifi = WiFiParams()
    bt = BluetoothParams()
    gps = GpsParams()
    spk = SpeakerParams()
    cellular = CellularParams()

    ctrl = ScreenOneStepOptimalController(
        screen=screen,
        weights=ScreenCostWeights(
            w_track=10.0,   # 保证“够亮”
            w_power=1.0,    # 惩罚功耗
            w_smooth=0.8,   # 避免频繁跳变
        ),
        update_interval_s=5.0,
        grid_points=51,
    )

    cpu_ctrl = CpuOneStepOptimalController(
        cpu=cpu,
        weights=CpuCostWeights(
            w_perf=35.0,     # 更重视算力满足
            w_power=0.9,     # 同时惩罚功耗
            w_smooth_f=0.6,
            w_smooth_n=0.2,
            w_smooth_u=0.1,
        ),
        update_interval_s=5.0,
        f_grid=7,
        n_grid=7,
    )

    cell_ctrl = CellularOneStepOptimalRateController(
        params=cellular,
        weights=CellularCostWeights(
            w_rate=2.0e-13,
            w_power=1.0,
            w_smooth=8.0e-13,
        ),
        update_interval_s=2.0,
        grid_points=21,
    )

    wifi_ctrl = WiFiOneStepOptimalRateController(
        params=wifi,
        weights=WiFiCostWeights(
            w_rate=1.0e-13,
            w_power=1.0,
            w_smooth=6.0e-13,
        ),
        update_interval_s=1.0,
    )

    bt_ctrl = BluetoothOneStepOptimalRateController(
        params=bt,
        weights=BluetoothCostWeights(
            w_rate=3.0e-13,
            w_power=1.0,
            w_smooth=1.0e-12,
        ),
        update_interval_s=1.0,
        grid_points=21,
    )

    gps_ctrl = GpsOneStepOptimalController(
        params=gps,
        weights=GpsCostWeights(
            w_update_deficit=2.0,
            w_acc_deficit=10.0,
            w_power=1.0,
            w_smooth_f=0.20,
            w_smooth_sigma=0.04,
        ),
        update_interval_s=1.0,
    )

    spk_ctrl = SpeakerOneStepOptimalVolumeController(
        params=spk,
        weights=SpeakerCostWeights(
            w_loud_deficit=10.0,
            w_power=1.0,
            w_smooth=0.8,
        ),
        update_interval_s=1.0,
        grid_points=21,
    )

    scenarios = [
        airplane_mode_reading(),
        piecewise_day_scenario(),
        worst_case_nav_hotspot(),
    ]

    print(f"\n=== {name}: 初始SOC z0={z0:.2f}（约{z0*100:.0f}%）, 环境温度 Tamb={t_amb_c:.1f}℃ ===")

    # 结果字段说明（只打印一次，后续各行中的字段均遵循此处含义与单位）
    print(
        "字段中文含义/单位说明：\n"
        "- 续航=预计电量耗尽时间（从当前SOC到z_min阈值）\n"
        "- 平均功耗Pavg(W)=系统总功耗时间平均值；括号内为各子系统平均功耗：屏幕/CPU/WiFi/BLE/GPS/扬声器/蜂窝\n"
        "- 亮度：Lreq=用户需求亮度(0~1)，Lcmd=控制输出亮度命令(0~1)\n"
        "- CPU：dem=需求负载(0~1)，f=频率(GHz)，n=激活核数，ueff=有效利用率(0~1)\n"
        "- GPS：f=更新率(Hz)，σreq=要求精度(m，越小越严格)，σest=估计精度(m)，Tmax=GPS芯片峰值温度(℃)\n"
        "- 扬声器：G=增益(0~1)，loud=响度估计(0~1)，Tmax=音圈峰值温度(℃)\n"
        "- WiFi/BLE/蜂窝：Rcmd=命令速率，Rsrv=实际服务速率，Q=队列长度，Tmax=芯片峰值温度\n"
        "- 热管理：Tmax(case/cpu/bat)=外壳/CPU/电池峰值温度(℃)，Heat(avg)=电池加热器平均‘输送热功率’(W)，elec=加热器平均电耗(W)，thr_min=全程最小节流因子(0~1，越小越强节流)\n"
        "- 三种方案：baseline=不进行优化控制(按需求直接驱动)；control=各组件单步MPC控制；planner=全时域滚动规划/离散模式MPC"
    )

    for sc in scenarios:
        # Baseline (requested brightness)
        dt_s = 2.0
        t_max_s = 24 * 3600

        t0, tr0 = simulate_soc(
            battery=battery,
            battery_soc=battery_soc,
            thermal=thermal,
            thermal_net=thermal_net,
            thermal_controller=thermal_controller,
            powers=powers,
            screen=screen,
            cpu=cpu,
            wifi=wifi,
            bluetooth=bt,
            gps=gps,
            speaker=spk,
            cellular=cellular,
            usage_fn=sc.usage_fn,
            screen_brightness_controller=None,
            cpu_controller=None,
            wifi_controller=None,
            bluetooth_controller=None,
            gps_controller=None,
            speaker_controller=None,
            cellular_controller=None,
            z0=z0,
            z_min=0.02,
            t_amb_c=t_amb_c,
            t0_c=t_amb_c,
            dt_s=dt_s,
            t_max_s=t_max_s,
        )

        # Optimal control (overrides brightness)
        ctrl.reset(l_cmd0=0.2)
        cpu_ctrl.reset()
        cell_ctrl.reset(r0_bps=0.0)
        wifi_ctrl.reset(r0_bps=0.0)
        bt_ctrl.reset(r0_bps=0.0)
        gps_ctrl.reset(f0_hz=0.0, sigma0_m=50.0)
        spk_ctrl.reset(g0=0.0)
        t1, tr1 = simulate_soc(
            battery=battery,
            battery_soc=battery_soc,
            thermal=thermal,
            thermal_net=thermal_net,
            thermal_controller=thermal_controller,
            powers=powers,
            screen=screen,
            cpu=cpu,
            wifi=wifi,
            bluetooth=bt,
            gps=gps,
            speaker=spk,
            cellular=cellular,
            usage_fn=sc.usage_fn,
            screen_brightness_controller=ctrl,
            cpu_controller=cpu_ctrl,
            wifi_controller=wifi_ctrl,
            bluetooth_controller=bt_ctrl,
            gps_controller=gps_ctrl,
            speaker_controller=spk_ctrl,
            cellular_controller=cell_ctrl,
            z0=z0,
            z_min=0.02,
            t_amb_c=t_amb_c,
            t0_c=t_amb_c,
            dt_s=dt_s,
            t_max_s=t_max_s,
        )

        # Full-horizon rollout / multi-segment MPC planner (discrete modes)
        t2, tr2 = plan_optimal_path(
            battery=battery,
            battery_soc=battery_soc,
            thermal=thermal,
            thermal_net=thermal_net,
            powers=powers,
            screen=screen,
            cpu=cpu,
            wifi=wifi,
            bluetooth=bt,
            gps=gps,
            speaker=spk,
            cellular=cellular,
            usage_fn=sc.usage_fn,
            z0=z0,
            z_min=0.02,
            t_amb_c=t_amb_c,
            t0_c=t_amb_c,
            dt_s=dt_s,
            rollout_dt_s=30.0,
            t_max_s=t_max_s,
            decision_interval_s=600.0,
            horizon_s=120.0,
            thermal_controller=thermal_controller,
        )

        def stats(tr):
            p_avg = sum(tr["p_sys_w"]) / max(1, len(tr["p_sys_w"]))
            p_scr_avg = sum(tr["p_screen_w"]) / max(1, len(tr["p_screen_w"]))
            p_cpu_avg = sum(tr["p_cpu_w"]) / max(1, len(tr["p_cpu_w"]))
            p_wifi_avg = sum(tr["p_wifi_w"]) / max(1, len(tr["p_wifi_w"]))
            p_bt_avg = sum(tr["p_bt_w"]) / max(1, len(tr["p_bt_w"]))
            p_gps_avg = sum(tr["p_gps_w"]) / max(1, len(tr["p_gps_w"]))
            p_spk_avg = sum(tr["p_spk_w"]) / max(1, len(tr["p_spk_w"]))
            p_cell_avg = sum(tr["p_cell_w"]) / max(1, len(tr["p_cell_w"]))
            l_req_avg = sum(tr["l_req"]) / max(1, len(tr["l_req"]))
            l_cmd_avg = sum(tr["l_cmd"]) / max(1, len(tr["l_cmd"]))
            d_avg = sum(tr["cpu_demand"]) / max(1, len(tr["cpu_demand"]))
            f_avg = sum(tr["cpu_f_ghz"]) / max(1, len(tr["cpu_f_ghz"]))
            n_avg = sum(tr["cpu_n_active"]) / max(1, len(tr["cpu_n_active"]))
            ueff_avg = sum(tr["cpu_u_eff"]) / max(1, len(tr["cpu_u_eff"]))
            r_cmd_avg = sum(tr["cell_r_cmd_bps"]) / max(1, len(tr["cell_r_cmd_bps"]))
            r_srv_avg = sum(tr["cell_r_served_bps"]) / max(1, len(tr["cell_r_served_bps"]))
            q_avg = sum(tr["cell_q_bits"]) / max(1, len(tr["cell_q_bits"]))
            t_modem_peak = max(tr["cell_t_modem_c"]) if tr["cell_t_modem_c"] else float("nan")

            w_cmd_avg = sum(tr["wifi_r_cmd_bps"]) / max(1, len(tr["wifi_r_cmd_bps"]))
            w_srv_avg = sum(tr["wifi_r_served_bps"]) / max(1, len(tr["wifi_r_served_bps"]))
            w_q_avg = sum(tr["wifi_q_bits"]) / max(1, len(tr["wifi_q_bits"]))
            w_t_peak = max(tr["wifi_t_c"]) if tr["wifi_t_c"] else float("nan")

            b_cmd_avg = sum(tr["bt_r_cmd_bps"]) / max(1, len(tr["bt_r_cmd_bps"]))
            b_srv_avg = sum(tr["bt_r_served_bps"]) / max(1, len(tr["bt_r_served_bps"]))
            b_q_avg = sum(tr["bt_q_bits"]) / max(1, len(tr["bt_q_bits"]))
            b_t_peak = max(tr["bt_t_c"]) if tr["bt_t_c"] else float("nan")

            g_f_avg = sum(tr["gps_f_update_hz"]) / max(1, len(tr["gps_f_update_hz"]))
            g_sigma_req_avg = sum(tr["gps_sigma_req_m"]) / max(1, len(tr["gps_sigma_req_m"]))
            g_sigma_est_avg = sum(tr["gps_sigma_est_m"]) / max(1, len(tr["gps_sigma_est_m"]))
            g_t_peak = max(tr["gps_t_c"]) if tr["gps_t_c"] else float("nan")

            spk_g_avg = sum(tr["spk_g_cmd"]) / max(1, len(tr["spk_g_cmd"]))
            spk_loud_avg = sum(tr["spk_loud_est"]) / max(1, len(tr["spk_loud_est"]))
            spk_t_peak = max(tr["spk_t_vc_c"]) if tr["spk_t_vc_c"] else float("nan")

            t_case_peak = max(tr.get("t_case_c", [float("nan")]))
            t_cpu_peak = max(tr.get("t_cpu_c", [float("nan")]))
            t_bat_peak = max(tr.get("t_bat_c", [float("nan")]))
            p_heat_avg = sum(tr.get("p_heat_w", [])) / max(1, len(tr.get("p_heat_w", [])))
            p_heat_elec_avg = sum(tr.get("p_heat_elec_w", [])) / max(1, len(tr.get("p_heat_elec_w", [])))
            throttle_min = min(tr.get("throttle_factor", [1.0])) if tr.get("throttle_factor") else 1.0
            return (
                p_avg,
                p_scr_avg,
                p_cpu_avg,
                p_wifi_avg,
                p_bt_avg,
                p_gps_avg,
                p_spk_avg,
                p_cell_avg,
                l_req_avg,
                l_cmd_avg,
                d_avg,
                f_avg,
                n_avg,
                ueff_avg,
                r_cmd_avg,
                r_srv_avg,
                q_avg,
                t_modem_peak,
                w_cmd_avg,
                w_srv_avg,
                w_q_avg,
                w_t_peak,
                b_cmd_avg,
                b_srv_avg,
                b_q_avg,
                b_t_peak,
                g_f_avg,
                g_sigma_req_avg,
                g_sigma_est_avg,
                g_t_peak,

                spk_g_avg,
                spk_loud_avg,
                spk_t_peak,

                t_case_peak,
                t_cpu_peak,
                t_bat_peak,
                p_heat_avg,
                p_heat_elec_avg,
                throttle_min,
            )

        (
            p0,
            ps0,
            pc0,
            pw0,
            pb0,
            pg0,
            pspk0,
            pcl0,
            lr0,
            lc0,
            d0,
            f0,
            n0,
            u0,
            rc0,
            rs0,
            q0,
            tm0,
            wc0,
            ws0,
            wq0,
            wt0,
            bc0,
            bs0,
            bq0,
            bt0,
            gf0,
            gsreq0,
            gsest0,
            gt0,
            sg0,
            sl0,
            st0,
            tc0,
            tcpu0,
            tbat0,
            ph0,
            phe0,
            thmin0,
        ) = stats(tr0)
        (
            p1,
            ps1,
            pc1,
            pw1,
            pb1,
            pg1,
            pspk1,
            pcl1,
            lr1,
            lc1,
            d1,
            f1,
            n1,
            u1,
            rc1,
            rs1,
            q1,
            tm1,
            wc1,
            ws1,
            wq1,
            wt1,
            bc1,
            bs1,
            bq1,
            bt1,
            gf1,
            gsreq1,
            gsest1,
            gt1,
            sg1,
            sl1,
            st1,
            tc1,
            tcpu1,
            tbat1,
            ph1,
            phe1,
            thmin1,
        ) = stats(tr1)
        (
            p2,
            ps2,
            pc2,
            pw2,
            pb2,
            pg2,
            pspk2,
            pcl2,
            lr2,
            lc2,
            d2,
            f2,
            n2,
            u2,
            rc2,
            rs2,
            q2,
            tm2,
            wc2,
            ws2,
            wq2,
            wt2,
            bc2,
            bs2,
            bq2,
            bt2,
            gf2,
            gsreq2,
            gsest2,
            gt2,
            sg2,
            sl2,
            st2,
            tc2,
            tcpu2,
            tbat2,
            ph2,
            phe2,
            thmin2,
        ) = stats(tr2)

        print(
            f"- {sc.name:22s} | baseline(基线): 续航={fmt_hours(t0):>10s} | 平均功耗Pavg={p0:5.2f}W "
            f"(屏幕{ps0:4.2f} CPU{pc0:4.2f} WiFi{pw0:4.2f} BLE{pb0:4.3f} GPS{pg0:4.3f} 扬声器{pspk0:4.3f} 蜂窝{pcl0:4.2f}) | "
            f"亮度(Lreq/Lcmd)={lr0:4.2f}/{lc0:4.2f} | CPU(dem,f,n,ueff)={d0:4.2f},{f0:4.2f}GHz,{n0:4.1f},{u0:4.2f} | "
            f"GPS(f,σreq,σest,Tmax)={gf0:4.2f}Hz,{gsreq0:5.1f}m,{gsest0:5.1f}m,{gt0:4.1f}℃ | "
            f"扬声器(G,loud,Tmax)={sg0:4.2f},{sl0:4.2f},{st0:4.1f}℃ | "
            f"WiFi(Rcmd/Rsrv,Q,Tmax)={wc0/1e6:5.2f}/{ws0/1e6:5.2f}Mbps,{wq0/1e6:5.2f}Mbit,{wt0:4.1f}℃ | "
            f"BLE(Rcmd/Rsrv,Q,Tmax)={bc0/1e3:5.1f}/{bs0/1e3:5.1f}kbps,{bq0/1e6:5.2f}Mbit,{bt0:4.1f}℃ | "
            f"蜂窝(Rcmd/Rsrv,Q,Tmax)={rc0/1e6:5.2f}/{rs0/1e6:5.2f}Mbps,{q0/1e6:5.2f}Mbit,{tm0:4.1f}℃ | "
            f"热管理Tmax(case/cpu/bat)=({tc0:4.1f}/{tcpu0:4.1f}/{tbat0:4.1f})℃ Heat(avg)={ph0:4.2f}W elec={phe0:4.2f}W thr_min={thmin0:4.2f}"
        )
        print(
            f"{'':26s} | control(单步MPC): 续航={fmt_hours(t1):>10s} | 平均功耗Pavg={p1:5.2f}W "
            f"(屏幕{ps1:4.2f} CPU{pc1:4.2f} WiFi{pw1:4.2f} BLE{pb1:4.3f} GPS{pg1:4.3f} 扬声器{pspk1:4.3f} 蜂窝{pcl1:4.2f}) | "
            f"亮度(Lreq/Lcmd)={lr1:4.2f}/{lc1:4.2f} | CPU(dem,f,n,ueff)={d1:4.2f},{f1:4.2f}GHz,{n1:4.1f},{u1:4.2f} | "
            f"GPS(f,σreq,σest,Tmax)={gf1:4.2f}Hz,{gsreq1:5.1f}m,{gsest1:5.1f}m,{gt1:4.1f}℃ | "
            f"扬声器(G,loud,Tmax)={sg1:4.2f},{sl1:4.2f},{st1:4.1f}℃ | "
            f"WiFi(Rcmd/Rsrv,Q,Tmax)={wc1/1e6:5.2f}/{ws1/1e6:5.2f}Mbps,{wq1/1e6:5.2f}Mbit,{wt1:4.1f}℃ | "
            f"BLE(Rcmd/Rsrv,Q,Tmax)={bc1/1e3:5.1f}/{bs1/1e3:5.1f}kbps,{bq1/1e6:5.2f}Mbit,{bt1:4.1f}℃ | "
            f"蜂窝(Rcmd/Rsrv,Q,Tmax)={rc1/1e6:5.2f}/{rs1/1e6:5.2f}Mbps,{q1/1e6:5.2f}Mbit,{tm1:4.1f}℃ | "
            f"热管理Tmax(case/cpu/bat)=({tc1:4.1f}/{tcpu1:4.1f}/{tbat1:4.1f})℃ Heat(avg)={ph1:4.2f}W elec={phe1:4.2f}W thr_min={thmin1:4.2f}"
        )

        mode_hint = ""
        if "planner_mode" in tr2 and tr2["planner_mode"]:
            # Most frequent mode
            from collections import Counter

            m = Counter(tr2["planner_mode"]).most_common(1)[0][0]
            mode_hint = f" mode={m}"

        print(
            f"{'':26s} | planner(全时域规划): 续航={fmt_hours(t2):>10s} | 平均功耗Pavg={p2:5.2f}W "
            f"(屏幕{ps2:4.2f} CPU{pc2:4.2f} WiFi{pw2:4.2f} BLE{pb2:4.3f} GPS{pg2:4.3f} 扬声器{pspk2:4.3f} 蜂窝{pcl2:4.2f}) | "
            f"亮度(Lreq/Lcmd)={lr2:4.2f}/{lc2:4.2f} | CPU(dem,f,n,ueff)={d2:4.2f},{f2:4.2f}GHz,{n2:4.1f},{u2:4.2f} | "
            f"GPS(f,σreq,σest,Tmax)={gf2:4.2f}Hz,{gsreq2:5.1f}m,{gsest2:5.1f}m,{gt2:4.1f}℃ | "
            f"扬声器(G,loud,Tmax)={sg2:4.2f},{sl2:4.2f},{st2:4.1f}℃ | "
            f"WiFi(Rcmd/Rsrv,Q,Tmax)={wc2/1e6:5.2f}/{ws2/1e6:5.2f}Mbps,{wq2/1e6:5.2f}Mbit,{wt2:4.1f}℃ | "
            f"BLE(Rcmd/Rsrv,Q,Tmax)={bc2/1e3:5.1f}/{bs2/1e3:5.1f}kbps,{bq2/1e6:5.2f}Mbit,{bt2:4.1f}℃ | "
            f"蜂窝(Rcmd/Rsrv,Q,Tmax)={rc2/1e6:5.2f}/{rs2/1e6:5.2f}Mbps,{q2/1e6:5.2f}Mbit,{tm2:4.1f}℃ | "
            f"热管理Tmax(case/cpu/bat)=({tc2:4.1f}/{tcpu2:4.1f}/{tbat2:4.1f})℃ Heat(avg)={ph2:4.2f}W elec={phe2:4.2f}W thr_min={thmin2:4.2f}{mode_hint}"
        )


def fast_thermal_check() -> None:
    """Quick smoke test to confirm heater/throttling signals are active."""

    from models.battery_model import (
        BatteryParams,
        BatterySocParams,
        BluetoothParams,
        CellularParams,
        ComponentPowers,
        CpuParams,
        GpsParams,
        ScreenParams,
        SpeakerParams,
        ThermalParams,
        ThermalNetworkParams,
        WiFiParams,
        simulate_soc,
    )
    from scenario_lib.scenarios import worst_case_nav_hotspot

    battery = BatteryParams(c_nom_ah=4.5, aging_alpha=0.10, eta_pmu=0.90, i_sd_a=0.0)
    battery_soc = BatterySocParams(q_nom_ah=battery.c_nom_ah, v_nom_v=0.5 * (battery.v_min + battery.v_max))

    thermal = ThermalParams(c_th_j_per_c=40.0, r_th_c_per_w=7.0)
    thermal_net = ThermalNetworkParams()
    thermal_controller = ThermalOneStepController(update_interval_s=5.0)

    powers = ComponentPowers()
    screen = ScreenParams()
    cpu = CpuParams()
    wifi = WiFiParams()
    bt = BluetoothParams()
    gps = GpsParams()
    spk = SpeakerParams()
    cellular = CellularParams()

    sc = worst_case_nav_hotspot()
    t_amb_c = 0.0
    t_empty_s, tr = simulate_soc(
        battery=battery,
        battery_soc=battery_soc,
        thermal=thermal,
        thermal_net=thermal_net,
        thermal_controller=thermal_controller,
        powers=powers,
        screen=screen,
        cpu=cpu,
        wifi=wifi,
        bluetooth=bt,
        gps=gps,
        speaker=spk,
        cellular=cellular,
        usage_fn=sc.usage_fn,
        z0=0.30,
        z_min=0.28,  # short segment
        t_amb_c=t_amb_c,
        t0_c=t_amb_c,
        dt_s=2.0,
        t_max_s=30 * 60,
    )

    t_case_peak = max(tr.get("t_case_c", [float("nan")]))
    t_cpu_peak = max(tr.get("t_cpu_c", [float("nan")]))
    t_bat_peak = max(tr.get("t_bat_c", [float("nan")]))
    p_heat_elec_avg = sum(tr.get("p_heat_elec_w", [])) / max(1, len(tr.get("p_heat_elec_w", [])))
    throttle_min = min(tr.get("throttle_factor", [1.0])) if tr.get("throttle_factor") else 1.0

    print("\n=== 快速热管理自检（FAST THERMAL CHECK） ===")
    print(f"续航（到阈值）={fmt_hours(t_empty_s)} | 环境温度Tamb={t_amb_c:.1f}℃")
    print(f"峰值温度Tmax(外壳/CPU/电池)=({t_case_peak:.1f}/{t_cpu_peak:.1f}/{t_bat_peak:.1f})℃")
    print(f"加热器平均电耗={p_heat_elec_avg:.2f}W | 全程最小节流因子={throttle_min:.2f}")


def main(argv: list[str] | None = None) -> None:
    """命令行入口。

    参数：
    - argv: 传入的命令行参数列表（不含程序名）。若为 None，则使用 sys.argv[1:]。
    """

    if argv is None:
        argv = sys.argv[1:]

    if "--print-traces-schema" in argv:
        print(traces_schema_markdown_cn())
        return

    write_schema_path = None
    for arg in argv:
        if arg.startswith("--write-traces-schema="):
            write_schema_path = arg.split("=", 1)[1].strip().strip('"')
            break
    if write_schema_path is None and "--write-traces-schema" in argv:
        idx = argv.index("--write-traces-schema")
        if idx + 1 < len(argv):
            write_schema_path = argv[idx + 1]

    if write_schema_path:
        with open(write_schema_path, "w", encoding="utf-8") as f:
            f.write(traces_schema_markdown_cn())
        print(f"已写入 traces 字段说明表：{write_schema_path}")
        return

    if "--fast-thermal" in argv:
        fast_thermal_check()
        return

    run_one(name="Warm day", z0=0.80, t_amb_c=25.0)
    run_one(name="Cold day", z0=0.80, t_amb_c=0.0)


if __name__ == "__main__":
    main()
