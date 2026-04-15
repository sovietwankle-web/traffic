"""疲劳驾驶员：反应慢、速度低且波动大、车道漂移、偶发微睡眠。"""

import random
from simulation.vehicle import DriverParams


def FatiguedParams() -> DriverParams:
    return DriverParams(
        desired_speed_factor=random.uniform(0.75, 0.9),
        min_gap=random.uniform(4.0, 8.0),
        time_headway=random.uniform(2.5, 4.0),
        max_accel=1.5,
        comfortable_decel=1.5,
        lane_change_threshold=0.5,
        politeness=0.5,
        reaction_delay=random.randint(8, 15),
        noise_std=0.6,
        red_light_run_prob=0.05,  # 偶尔因反应慢闯红灯
        monitor_awareness=0.0,
        lane_drift_amplitude=random.uniform(0.3, 0.5),
        micro_sleep_prob=0.001,   # 每tick 0.1%概率，约100秒触发一次
        micro_sleep_duration=random.randint(10, 30),  # 1-3秒
    )
