from __future__ import annotations

from dataclasses import dataclass


def clamp01(x: float) -> float:
    """把数值夹紧到 [0,1]。"""

    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


@dataclass(frozen=True)
class UsageInputs:
    """场景/应用给出的“需求侧输入”。

    说明：
    - 为了兼容已有场景（scenario_lib/scenarios.py），这里保留了较多字段。
    - 简化温度耦合模型只会用到其中一小部分：brightness, cpu_demand,
      wifi_rate_bps, cell_arrival_bps, bt_arrival_bps, gps_update_min_hz,
      gps_sigma_max_m, speaker_volume。
    """

    # Normalized controls / rates
    brightness: float  # L_cmd in [0,1]

    # CPU workload demand (normalized required compute)
    cpu_demand: float  # in [0,1]

    # Screen controls (kept for scenario compatibility)
    screen_gamma: float
    screen_refresh_hz: float
    screen_res_scale: float
    screen_active_area: float

    # Legacy GPS knobs (kept)
    gps_fix_duty: float
    gps_acq_rate_hz: float

    # WiFi offered load
    wifi_rate_bps: float
    wifi_scan_rate_hz: float

    # Cellular offered load
    cell_arrival_bps: float

    # Bluetooth offered load
    bt_arrival_bps: float

    bt_on: float

    speaker_volume: float

    # GPS (new-ish requirement fields)
    gps_update_min_hz: float = 0.0
    gps_sigma_max_m: float = 50.0
    gps_on: float = 0.0
    gps_cn0_env_dbhz: float = 35.0
    gps_n_vis_raw: float = 8.0
    gps_cn0_thresh_dbhz: float = 28.0
    gps_b_loop_hz: float = 8.0
    gps_assist: float = 0.0

    # Speaker (kept)
    spk_audio_level: float = 0.0
    spk_v_limit_v: float = 1.0
    spk_mode: int = 0

    # Ambient temperature (optional; some scenarios may set)
    ambient_c: float = 25.0
