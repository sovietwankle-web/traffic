"""普通人类驾驶员：参数有随机波动，有反应延迟和噪声。"""

import random
from simulation.vehicle import DriverParams


def NormalHumanParams() -> DriverParams:
    return DriverParams(
        desired_speed_factor=random.uniform(0.9, 1.05),
        min_gap=random.uniform(3.0, 5.0),
        time_headway=random.uniform(1.5, 2.5),
        max_accel=random.uniform(2.0, 3.0),
        comfortable_decel=random.uniform(2.0, 3.0),
        lane_change_threshold=random.uniform(0.15, 0.3),
        politeness=random.uniform(0.3, 0.7),
        reaction_delay=random.randint(2, 5),
        noise_std=0.3,
        red_light_run_prob=0.0,
        monitor_awareness=0.0,
        lane_drift_amplitude=0.0,
        micro_sleep_prob=0.0,
        micro_sleep_duration=0,
    )
