"""违规追速者：大幅超速、频繁变道、闯红灯，在已知监测点附近会减速伪装。"""

import random
from simulation.vehicle import DriverParams


def RuleBreakerParams() -> DriverParams:
    return DriverParams(
        desired_speed_factor=random.uniform(1.15, 1.35),
        min_gap=random.uniform(1.5, 2.5),
        time_headway=random.uniform(0.8, 1.2),
        max_accel=4.0,
        comfortable_decel=4.0,
        lane_change_threshold=0.05,
        politeness=0.05,
        reaction_delay=1,
        noise_std=0.2,
        red_light_run_prob=0.2,
        monitor_awareness=0.3,  # 30%概率感知到监测点并减速
        lane_drift_amplitude=0.0,
        micro_sleep_prob=0.0,
        micro_sleep_duration=0,
    )
