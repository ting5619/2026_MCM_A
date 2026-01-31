from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Callable

from models.battery_model import UsageInputs
from models.simplified_model import SimplifiedThermalBatteryParams, simulate
from scenario_lib import scenarios


def _get_scenario(name: str) -> scenarios.Scenario:
    mapping = {
        "piecewise_day": scenarios.piecewise_day_scenario,
        "worst_case_nav_hotspot": scenarios.worst_case_nav_hotspot,
        "airplane_mode_reading": scenarios.airplane_mode_reading,
    }
    if name not in mapping:
        raise SystemExit(f"未知场景: {name}. 可选: {', '.join(mapping)}")
    return mapping[name]()


def _wrap_usage_with_ambient(usage_fn: Callable[[float], UsageInputs], t_amb_c: float) -> Callable[[float], UsageInputs]:
    """把环境温度写回 UsageInputs（如果该字段存在）。"""

    def f(t_s: float) -> UsageInputs:
        u = usage_fn(t_s)
        # 旧模型/场景里可能带 ambient_c 字段；没有也不影响。
        if hasattr(u, "ambient_c"):
            setattr(u, "ambient_c", float(t_amb_c))
        return u

    return f


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "简化温度耦合模型仿真（SOC+系统温度两状态）。\n"
            "推荐运行：python -m scripts.simulate_simplified --scenario piecewise_day"
        )
    )
    parser.add_argument("--scenario", default="piecewise_day", help="场景名")
    parser.add_argument("--t-amb", type=float, default=25.0, help="环境温度 T_amb (℃)")
    parser.add_argument("--t0", type=float, default=30.0, help="初始系统温度 T_sys0 (℃)")
    parser.add_argument("--z0", type=float, default=1.0, help="初始 SOC z0 (0~1)")
    parser.add_argument("--dt", type=float, default=1.0, help="积分步长 dt (s)")
    parser.add_argument("--t-max", type=float, default=6 * 3600.0, help="最长仿真时间 (s)")
    parser.add_argument("--no-control", action="store_true", help="关闭温度/SOC 感知控制（直接按需求运行）")
    parser.add_argument("--compare", action="store_true", help="同时跑‘有控制’与‘无控制’两次并对比")
    parser.add_argument("--print-params", action="store_true", help="打印模型参数")

    args = parser.parse_args(argv)

    sc = _get_scenario(args.scenario)
    usage_fn = _wrap_usage_with_ambient(sc.usage_fn, t_amb_c=args.t_amb)

    params = SimplifiedThermalBatteryParams()

    if args.print_params:
        print("=== SimplifiedThermalBatteryParams ===")
        for k, v in asdict(params).items():
            print(f"{k}: {v}")
        print()

    def run_once(use_control: bool) -> float:
        t_empty_s, traces = simulate(
            params=params,
            usage_fn=usage_fn,
            z0=args.z0,
            t0_sys_c=args.t0,
            t_amb_c=args.t_amb,
            dt_s=args.dt,
            t_max_s=args.t_max,
            use_control=use_control,
        )
        # 简单摘要
        t_end = traces["t_s"][-1] if traces["t_s"] else 0.0
        soc_end = traces["soc"][-1] if traces["soc"] else args.z0
        t_sys_end = traces["t_sys_c"][-1] if traces["t_sys_c"] else args.t0
        t_sys_peak = max(traces["t_sys_c"], default=args.t0)
        p_peak = max(traces["p_total_w"], default=0.0)

        label = "温度/SOC控制=开" if use_control else "温度/SOC控制=关"
        print(f"=== {label} ===")
        print(f"场景: {sc.name}")
        print(f"t_empty: {t_empty_s/60.0:.2f} min (仿真结束 t={t_end/60.0:.2f} min)")
        print(f"SOC_end: {soc_end:.4f}")
        print(f"T_sys_end: {t_sys_end:.2f} ℃, T_sys_peak: {t_sys_peak:.2f} ℃")
        print(f"P_total_peak: {p_peak:.2f} W")
        print()
        return t_empty_s

    if args.compare:
        t1 = run_once(use_control=True)
        t0 = run_once(use_control=False)
        if t0 > 0:
            print(f"对比：控制开启续航提升 = {(t1 - t0) / 60.0:.2f} min")
        return 0

    use_control = not args.no_control
    run_once(use_control=use_control)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
