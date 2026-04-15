"""遵守规则的AI车：严格遵守所有交规，零噪声，即时反应。"""

import random
from simulation.vehicle import DriverParams


def RuleFollowerParams() -> DriverParams:
    return DriverParams(
        desired_speed_factor=random.uniform(0.95, 1.0),
        min_gap=5.0,
        time_headway=2.0,
        max_accel=2.0,
        comfortable_decel=2.5,
        lane_change_threshold=0.3,
        politeness=0.8,
        reaction_delay=0,
        noise_std=0.0,
        red_light_run_prob=0.0,
        monitor_awareness=0.0,
        lane_drift_amplitude=0.0,
        micro_sleep_prob=0.0,
        micro_sleep_duration=0,
    )
