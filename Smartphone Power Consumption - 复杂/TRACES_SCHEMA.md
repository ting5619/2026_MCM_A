# traces 字典字段说明（键名 → 中文含义/单位）

说明：simulate_soc() 与 path_planner.plan_optimal_path() 返回的 traces 字典字段含义如下。

| 键名 | 中文含义 | 单位 | 备注 |
|---|---|---|---|
| t_s | 仿真时间 | s | 从 0 开始累计的模拟时间 |
| soc | 电池荷电状态SOC（比例） | 1 | 范围 0~1 |
| soc_pct | 电池荷电状态SOC（百分比） | % | =100*soc |
| temp_c | 手机整体温度代理（与 t_case_c 当前一致） | ℃ | 无三节点热网时为一阶热模型温度；有三节点热网时等于外壳温度 |
| t_case_c | 外壳/机身温度 | ℃ | 当前实现中与 temp_c 同值 |
| t_bat_c | 电池温度 | ℃ | 三节点热网时为电池节点温度；否则来自电池温度代数模型(平滑) |
| q_max_ah | 电池最大可用容量（考虑老化/温度修正） | Ah | advanced battery model 开启时会随时间衰减 |
| v_oc_v | 电池开路电压（OCV） | V | 由 SOC→OCV 模型计算 |
| i_bat_a | 电池侧放电电流 | A | 用于 SOC 递推的电流；正值表示放电 |
| r_bat_ohm | 电池等效内阻 | Ω | 由 SOC/温度模型得到的等效内阻 |
| p_device_w | 设备负载功耗（不含电池内部损耗） | W | 各子系统功耗 + 基线功耗 +（若启用）加热器电耗 |
| p_loss_w | 电池内部损耗功率 | W | 内阻损耗 + 放电效率等效损耗 |
| p_heat_w | 电池加热器‘输送到电池’的热功率 | W | 热管理控制输出；与 p_heat_elec_w 不同 |
| p_heat_elec_w | 加热器从电池消耗的电功率 | W | = p_heat_w / η_heat(T) 的近似 |
| throttle_factor | 热节流因子 | 1 | 0~1；越小表示越强的性能/需求压缩 |
| t_screen_c | 屏幕温度 | ℃ | ScreenState.t_s_c |
| t_cpu_c | CPU结温/核心温度 | ℃ | CpuState.t_j_c（若三节点热网启用则与CPU节点对齐） |
| p_sys_w | 系统从电池侧抽取的总功率（SOC更新用） | W | 通常约等于 p_device_w + p_loss_w |
| p_screen_w | 屏幕功耗 | W | P_screen = backlight + driver + leakage |
| p_cpu_w | CPU功耗 | W | P_cpu = dynamic + static + clock |
| p_wifi_w | WiFi模块功耗 | W | 含PA/BB/RF/时钟/泄漏等 |
| p_bt_w | 蓝牙(BLE)模块功耗 | W | 连接事件平均化模型输出 |
| p_gps_w | GPS模块功耗 | W | 含RF/相关器/DSP/存储/时钟/泄漏等 |
| p_spk_w | 扬声器/功放功耗 | W | Class-D效率曲线 + 音圈焦耳热等效 |
| p_cell_w | 蜂窝基带/射频模块功耗 | W | 含PA/BB/RF等 |
| l_req | 用户请求亮度（场景输入） | 1 | 0~1 |
| l_cmd | 控制器输出的亮度命令 | 1 | 0~1；可能不同于 l_req |
| l_eff | 屏幕有效亮度（考虑响应时间常数） | 1 | 0~1；用于功耗与体验评估 |
| cpu_demand | CPU需求负载（场景输入） | 1 | 0~1；表示应用对算力的需求 |
| cpu_f_ghz | CPU频率命令 | GHz | 控制器输出（DVFS） |
| cpu_n_active | CPU激活核心数 | 个 | 0..n_total |
| cpu_u_eff | CPU有效利用率 | 1 | 0~1；考虑动态响应后的实际利用率 |
| cpu_u_cmd | CPU命令利用率 | 1 | 0~1；控制器输出 |
| wifi_r_cmd_bps | WiFi命令速率 | bps | 控制器输出 |
| wifi_r_served_bps | WiFi实际服务速率 | bps | 受链路容量/状态机影响 |
| wifi_q_bits | WiFi队列长度 | bit | 到达-服务的队列积累 |
| wifi_t_c | WiFi芯片温度 | ℃ | 热模型输出 |
| wifi_p_tx_w | WiFi发射功率 | W | 功率控制的一阶惯性状态 |
| wifi_p_tx_state | WiFi处于TX状态的概率p5 | 1 | WiFiState.p5（CTMC概率） |
| bt_r_cmd_bps | BLE命令速率 | bps | 控制器输出 |
| bt_r_served_bps | BLE实际服务速率 | bps | 由连接事件占空比与PER等决定 |
| bt_q_bits | BLE队列长度 | bit | 到达-服务的队列积累 |
| bt_t_c | BLE芯片温度 | ℃ | 热模型输出 |
| bt_per | BLE误包率PER | 1 | 0~1；随RSSI平滑变化 |
| bt_rssi_dbm | BLE平均RSSI | dBm | RSSI平滑估计 |
| bt_p_tx_dbm | BLE发射功率（dBm） | dBm | 自适应功率控制状态（离散档位的平滑近似） |
| gps_f_update_hz | GPS更新率命令 | Hz | 控制器输出；0表示关闭 |
| gps_sigma_req_m | GPS精度要求（越小越严格） | m | 控制器输出 |
| gps_sigma_est_m | GPS当前精度估计 | m | 模型状态 |
| gps_cn0_dbhz | GPS载噪比C/N0 | dB-Hz | 环境与动态共同决定 |
| gps_n_locked | GPS锁定卫星数（代理） | 颗 | 可为连续值（平滑建模） |
| gps_lq | GPS锁定质量LQ | 1 | 0~1 |
| gps_t_c | GPS芯片温度 | ℃ | 热模型输出 |
| gps_m_off | GPS模式概率：OFF | 1 | 0~1 |
| gps_m_standby | GPS模式概率：STANDBY | 1 | 0~1 |
| gps_m_acq | GPS模式概率：ACQUISITION | 1 | 0~1 |
| gps_m_track | GPS模式概率：TRACKING | 1 | 0~1 |
| gps_m_nav | GPS模式概率：NAVIGATION | 1 | 0~1 |
| gps_m_assist | GPS模式概率：ASSIST | 1 | 0~1 |
| spk_g_cmd | 扬声器数字增益命令 | 1 | 0~1；控制器输出 |
| spk_i_vc_a | 音圈电流 | A | 包络/平均化模型状态 |
| spk_v_filter_v | 输出滤波电压 | V | LC输出滤波器等效电压 |
| spk_t_vc_c | 音圈温度 | ℃ | 热模型输出 |
| spk_p_joule_w | 音圈焦耳热功率 | W | = I_vc^2 * R_vc(T) |
| spk_loud_est | 响度估计（代理） | 1 | 0~1；由功率映射得到，便于控制 |
| cell_r_cmd_bps | 蜂窝命令速率 | bps | 控制器输出 |
| cell_r_served_bps | 蜂窝实际服务速率 | bps | 受链路容量/状态机影响 |
| cell_q_bits | 蜂窝数据队列长度 | bit | 到达-服务的队列积累 |
| cell_t_modem_c | 蜂窝调制解调器温度 | ℃ | 热模型输出 |
| cell_p_tx_w | 蜂窝发射功率 | W | 功率控制状态 |
| cell_p_high | 蜂窝处于高活跃状态的概率p3 | 1 | CellularState.p3（HIGH_ACTIVITY概率） |
| planner_mode | 规划器选择的离散模式 | - | eco / balanced / perf；仅在 planner traces 中出现 |
