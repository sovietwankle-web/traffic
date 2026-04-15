"""用户需求至上的AI车：总体遵守规则，但激进换道、快速加速、跟车距离近。"""

import random
from simulation.vehicle import DriverParams


def AggressiveAIParams() -> DriverParams:
    return DriverParams(
        desired_speed_factor=random.uniform(1.0, 1.05),
        min_gap=2.5,
        time_headway=1.2,
        max_accel=3.5,
        comfortable_decel=3.0,
        lane_change_threshold=0.1,
        politeness=0.2,
        reaction_delay=0,
        noise_std=0.05,
        red_light_run_prob=0.0,
        monitor_awareness=0.0,
        lane_drift_amplitude=0.0,
        micro_sleep_prob=0.0,
        micro_sleep_duration=0,
    )
