"""
场景5：都市六出口环岛
环形单车道，限速30km/h，周长约200m
6条进近道，各2车道，限速50km/h
让行进入（无信号灯）
"""

from simulation.road_network import (
    RoadNetwork, RoadSegment, Intersection, Connection,
)
from simulation.monitor import MonitorSystem, MonitoringPoint
from simulation.engine import SimulationEngine
from simulation.drivers.base import create_driver_factory, DEFAULT_DRIVER_WEIGHTS


SCENARIO_NAME = "roundabout"
SCENARIO_DESC = "都市六出口环岛(环内限速30km/h)"


def build_roundabout(seed=None) -> SimulationEngine:
    network = RoadNetwork()

    circ_limit = 30.0 / 3.6   # 8.33 m/s
    approach_limit = 50.0 / 3.6  # 13.89 m/s

    # 6个进近道（入口段）
    for i in range(6):
        network.add_segment(RoadSegment(f"approach_{i}", 150, 2, approach_limit, "urban"))
        network.add_segment(RoadSegment(f"depart_{i}", 150, 2, approach_limit, "urban"))

    # 环形道分为6段（两个相邻出入口之间一段）
    circ_length = 200.0 / 6  # 约33m每段
    for i in range(6):
        network.add_segment(RoadSegment(
            f"circ_{i}", circ_length, 1, circ_limit, "urban", curvature=0.5
        ))

    # 6个汇入/分出点
    for i in range(6):
        network.add_intersection(Intersection(f"node_{i}"))

    # 连接
    for i in range(6):
        next_i = (i + 1) % 6

        # 环内直行：circ_i -> circ_next
        network.add_connection(Connection(f"circ_{i}", f"circ_{next_i}", f"node_{next_i}", "straight", 3.0))

        # 进近道 -> 环内
        network.add_connection(Connection(f"approach_{i}", f"circ_{i}", f"node_{i}", "merge", 1.5))

        # 环内 -> 离开道
        network.add_connection(Connection(f"circ_{i}", f"depart_{i}", f"node_{i}", "diverge", 1.0))

    network.build_index()

    # 监测点
    ms = MonitorSystem()
    for i in range(6):
        ms.add_monitor(MonitoringPoint(f"rm_app_{i}", f"approach_{i}", 100.0, f"环岛进口{i+1}"))
        ms.add_monitor(MonitoringPoint(f"rm_circ_{i}", f"circ_{i}", circ_length * 0.5, f"环岛段{i+1}"))

    engine = SimulationEngine(network, ms, dt=0.1, seed=seed)
    engine.set_driver_factory(create_driver_factory())

    for i in range(6):
        engine.add_spawn_config(f"approach_{i}", 0.15, DEFAULT_DRIVER_WEIGHTS)

    return engine
