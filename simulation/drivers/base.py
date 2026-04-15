"""驾驶员工厂：根据类型创建参数。"""

from typing import Tuple, Callable

from simulation.vehicle import (
    DriverParams, DRIVER_RULE_FOLLOWER, DRIVER_AGGRESSIVE_AI,
    DRIVER_NORMAL_HUMAN, DRIVER_RULE_BREAKER, DRIVER_FATIGUED,
)
from simulation.road_network import RoadSegment
from simulation.drivers.rule_follower import RuleFollowerParams
from simulation.drivers.aggressive_ai import AggressiveAIParams
from simulation.drivers.normal_human import NormalHumanParams
from simulation.drivers.rule_breaker import RuleBreakerParams
from simulation.drivers.fatigued import FatiguedParams


def create_driver_params(driver_type: str) -> DriverParams:
    """根据驾驶员类型创建参数"""
    factories = {
        DRIVER_RULE_FOLLOWER: RuleFollowerParams,
        DRIVER_AGGRESSIVE_AI: AggressiveAIParams,
        DRIVER_NORMAL_HUMAN: NormalHumanParams,
        DRIVER_RULE_BREAKER: RuleBreakerParams,
        DRIVER_FATIGUED: FatiguedParams,
    }
    factory = factories.get(driver_type)
    if factory is None:
        raise ValueError(f"Unknown driver type: {driver_type}")
    return factory()


def create_driver_factory() -> Callable:
    """创建driver_factory供SimulationEngine使用。
    返回: callable(driver_type, segment) -> (DriverParams, initial_speed)
    """
    def factory(driver_type: str, segment: RoadSegment) -> Tuple[DriverParams, float]:
        params = create_driver_params(driver_type)
        # 初始速度：期望速度的80-100%
        desired = segment.speed_limit * params.desired_speed_factor
        initial_speed = desired * 0.9
        return params, initial_speed

    return factory


# 默认的驾驶员类型权重分布
DEFAULT_DRIVER_WEIGHTS = {
    DRIVER_RULE_FOLLOWER: 0.25,
    DRIVER_AGGRESSIVE_AI: 0.15,
    DRIVER_NORMAL_HUMAN: 0.30,
    DRIVER_RULE_BREAKER: 0.15,
    DRIVER_FATIGUED: 0.15,
}
