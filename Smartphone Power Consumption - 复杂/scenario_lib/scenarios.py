from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from models.battery_model import UsageInputs


@dataclass(frozen=True)
class Scenario:
    name: str
    usage_fn: Callable[[float], UsageInputs]


def piecewise_day_scenario() -> Scenario:
    """一个“通勤+办公+午休+晚间娱乐”的分段常数场景。"""

    def usage(t_s: float) -> UsageInputs:
        t_h = (t_s / 3600.0) % 24.0

        # Default: light standby
        brightness = 0.08
        screen_gamma = 2.2
        screen_refresh_hz = 60.0
        screen_res_scale = 1.0
        screen_active_area = 0.6
        cpu_demand = 0.06
        gps_fix_duty = 0.0
        gps_acq_rate = 0.0

        # GPS (new model): default off
        gps_on = 0.0
        gps_update_min_hz = 0.0
        gps_sigma_max_m = 50.0
        gps_cn0_env_dbhz = 35.0
        gps_n_vis_raw = 8.0
        gps_cn0_thresh_dbhz = 28.0
        gps_b_loop_hz = 8.0
        gps_assist = 0.0
        wifi_rate = 0.0
        wifi_scan_rate = 1 / 120.0
        cell_arrival_bps = 30e3
        bt_arrival_bps = 8e3
        bt_on = 0.5
        spk_vol = 0.0
        spk_audio_level = 0.0
        spk_v_limit_v = 1.0
        spk_mode = 0

        # 7-9 commute: navigation + screen on, cellular and GPS
        if 7 <= t_h < 9:
            brightness = 0.6
            screen_refresh_hz = 60.0
            screen_active_area = 1.0
            cpu_demand = 0.35
            gps_on = 1.0
            gps_update_min_hz = 1.0
            gps_sigma_max_m = 10.0
            gps_cn0_env_dbhz = 32.0
            gps_n_vis_raw = 9.0
            gps_assist = 0.3

            # keep legacy knobs at zero (avoid double meaning)
            gps_fix_duty = 0.0
            gps_acq_rate = 0.0
            wifi_rate = 0.0
            cell_arrival_bps = 350e3
            bt_arrival_bps = 10e3

        # 9-12 office: screen moderate, WiFi active
        if 9 <= t_h < 12:
            brightness = 0.35
            screen_refresh_hz = 60.0
            screen_active_area = 0.9
            cpu_demand = 0.22
            wifi_rate = 2e6  # 2 Mbps
            wifi_scan_rate = 1 / 300.0
            cell_arrival_bps = 80e3
            bt_arrival_bps = 25e3

            gps_on = 0.0

        # 12-13 lunch: video on WiFi
        if 12 <= t_h < 13:
            brightness = 0.55
            screen_refresh_hz = 60.0
            screen_active_area = 1.0
            cpu_demand = 0.40
            wifi_rate = 8e6
            wifi_scan_rate = 1 / 600.0
            spk_vol = 0.3
            spk_audio_level = 1.0
            spk_mode = 1
            cell_arrival_bps = 40e3
            bt_arrival_bps = 15e3

            gps_on = 0.0

        # 13-18 afternoon: mostly idle + sporadic network
        if 13 <= t_h < 18:
            brightness = 0.25
            screen_refresh_hz = 60.0
            screen_active_area = 0.8
            cpu_demand = 0.16
            wifi_rate = 1e6
            wifi_scan_rate = 1 / 300.0
            cell_arrival_bps = 60e3
            bt_arrival_bps = 12e3

            gps_on = 0.0

        # 20-23 evening gaming on WiFi
        if 20 <= t_h < 23:
            brightness = 0.75
            screen_refresh_hz = 120.0
            screen_active_area = 1.0
            cpu_demand = 0.80
            wifi_rate = 5e6
            wifi_scan_rate = 1 / 900.0
            bt_on = 0.7
            cell_arrival_bps = 50e3
            bt_arrival_bps = 35e3

            gps_on = 0.0

        return UsageInputs(
            brightness=brightness,
            cpu_demand=cpu_demand,

            screen_gamma=screen_gamma,
            screen_refresh_hz=screen_refresh_hz,
            screen_res_scale=screen_res_scale,
            screen_active_area=screen_active_area,

            gps_fix_duty=gps_fix_duty,
            gps_acq_rate_hz=gps_acq_rate,

            gps_update_min_hz=gps_update_min_hz,
            gps_sigma_max_m=gps_sigma_max_m,
            gps_on=gps_on,
            gps_cn0_env_dbhz=gps_cn0_env_dbhz,
            gps_n_vis_raw=gps_n_vis_raw,
            gps_cn0_thresh_dbhz=gps_cn0_thresh_dbhz,
            gps_b_loop_hz=gps_b_loop_hz,
            gps_assist=gps_assist,

            wifi_rate_bps=wifi_rate,
            wifi_scan_rate_hz=wifi_scan_rate,
            cell_arrival_bps=cell_arrival_bps,
            bt_arrival_bps=bt_arrival_bps,
            bt_on=bt_on,
            speaker_volume=spk_vol,

            spk_audio_level=spk_audio_level,
            spk_v_limit_v=spk_v_limit_v,
            spk_mode=spk_mode,
        )

    return Scenario(name="piecewise_day", usage_fn=usage)


def worst_case_nav_hotspot() -> Scenario:
    """高亮+GPS常开+蜂窝热点传输：典型快速掉电场景。"""

    def usage(_: float) -> UsageInputs:
        return UsageInputs(
            brightness=0.9,
            cpu_demand=0.55,

            screen_gamma=2.2,
            screen_refresh_hz=60.0,
            screen_res_scale=1.0,
            screen_active_area=1.0,

            gps_fix_duty=0.0,
            gps_acq_rate_hz=0.0,

            gps_update_min_hz=5.0,
            gps_sigma_max_m=5.0,
            gps_on=1.0,
            gps_cn0_env_dbhz=29.0,
            gps_n_vis_raw=7.0,
            gps_cn0_thresh_dbhz=30.0,
            gps_b_loop_hz=12.0,
            gps_assist=0.6,

            wifi_rate_bps=0.0,
            wifi_scan_rate_hz=0.0,
            cell_arrival_bps=12e6,
            bt_arrival_bps=50e3,
            bt_on=0.3,
            speaker_volume=0.0,

            spk_audio_level=0.0,
            spk_v_limit_v=1.0,
            spk_mode=0,
        )

    return Scenario(name="worst_case_nav_hotspot", usage_fn=usage)


def airplane_mode_reading() -> Scenario:
    """飞行模式阅读：低CPU+低网络+中等亮度。"""

    def usage(_: float) -> UsageInputs:
        return UsageInputs(
            brightness=0.35,
            cpu_demand=0.10,

            screen_gamma=2.2,
            screen_refresh_hz=60.0,
            screen_res_scale=0.85,
            screen_active_area=1.0,

            gps_fix_duty=0.0,
            gps_acq_rate_hz=0.0,

            gps_update_min_hz=0.0,
            gps_sigma_max_m=80.0,
            gps_on=0.0,
            gps_cn0_env_dbhz=35.0,
            gps_n_vis_raw=8.0,
            gps_cn0_thresh_dbhz=28.0,
            gps_b_loop_hz=8.0,
            gps_assist=0.0,

            wifi_rate_bps=0.0,
            wifi_scan_rate_hz=0.0,
            cell_arrival_bps=0.0,
            bt_arrival_bps=0.0,
            bt_on=0.0,
            speaker_volume=0.0,

            spk_audio_level=0.0,
            spk_v_limit_v=1.0,
            spk_mode=0,
        )

    return Scenario(name="airplane_mode_reading", usage_fn=usage)
