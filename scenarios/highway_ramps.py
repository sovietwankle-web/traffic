"""
场景2：高速公路 + 3个先下后上的出入口
主路：3车道，限速120km/h，总长约3500m
每个出入口有300m匝道，限速60km/h
"""

from simulation.road_network import (
    RoadNetwork, RoadSegment, Intersection, Connection,
)
from simulation.monitor import MonitorSystem, MonitoringPoint
from simulation.engine import SimulationEngine
from simulation.drivers.base import create_driver_factory, DEFAULT_DRIVER_WEIGHTS


SCENARIO_NAME = "highway_ramps"
SCENARIO_DESC = "高速公路三出入口(限速120km/h)"


def build_highway_ramps(seed=None) -> SimulationEngine:
    network = RoadNetwork()

    hw_limit = 120.0 / 3.6   # 33.33 m/s
    ramp_limit = 60.0 / 3.6  # 16.67 m/s

    # 主路分段
    network.add_segment(RoadSegment("hw_entry", 500, 3, hw_limit, "highway"))
    network.add_segment(RoadSegment("hw_1", 800, 3, hw_limit, "highway"))
    network.add_segment(RoadSegment("hw_2", 800, 3, hw_limit, "highway"))
    network.add_segment(RoadSegment("hw_3", 800, 3, hw_limit, "highway"))
    network.add_segment(RoadSegment("hw_exit", 500, 3, hw_limit, "highway"))

    # 匝道（先下后上：出口匝道 + 入口匝道）
    for i in range(1, 4):
        network.add_segment(RoadSegment(f"off_ramp_{i}", 300, 1, ramp_limit, "ramp"))
        network.add_segment(RoadSegment(f"on_ramp_{i}", 300, 1, ramp_limit, "ramp"))

    # 交叉口（汇流/分流点）- 无信号灯
    for i in range(1, 4):
        # 分流点
        div_id = f"diverge_{i}"
        network.add_intersection(Intersection(div_id, [f"hw_{i-1}" if i > 1 else "hw_entry"], [f"hw_{i}", f"off_ramp_{i}"]))
        # 汇流点
        merge_id = f"merge_{i}"
        next_hw = f"hw_{i+1}" if i < 3 else "hw_exit"
        network.add_intersection(Intersection(merge_id, [f"hw_{i}", f"on_ramp_{i}"], [next_hw]))

    # 连接 - 主路直行
    network.add_connection(Connection("hw_entry", "hw_1", "diverge_1", "straight", 5.0))
    network.add_connection(Connection("hw_1", "hw_2", "merge_1", "straight", 5.0))
    # 需要添加merge到下一个diverge的直行（hw_2经过diverge_2）
    network.add_connection(Connection("hw_2", "hw_3", "merge_2", "straight", 5.0))
    network.add_connection(Connection("hw_3", "hw_exit", "merge_3", "straight", 5.0))

    # 分流连接
    network.add_connection(Connection("hw_entry", "off_ramp_1", "diverge_1", "diverge", 0.3))
    network.add_connection(Connection("hw_1", "off_ramp_2", "diverge_2", "diverge", 0.3))
    network.add_connection(Connection("hw_2", "off_ramp_3", "diverge_3", "diverge", 0.3))

    # 汇流连接
    network.add_connection(Connection("on_ramp_1", "hw_2", "merge_1", "merge", 1.0))
    network.add_connection(Connection("on_ramp_2", "hw_3", "merge_2", "merge", 1.0))
    network.add_connection(Connection("on_ramp_3", "hw_exit", "merge_3", "merge", 1.0))

    network.build_index()

    # 监测点
    ms = MonitorSystem()
    ms.add_monitor(MonitoringPoint("hm1", "hw_entry", 250.0, "高速入口"))
    ms.add_monitor(MonitoringPoint("hm2", "hw_1", 400.0, "匝道1前"))
    ms.add_monitor(MonitoringPoint("hm3", "hw_2", 400.0, "匝道2前"))
    ms.add_monitor(MonitoringPoint("hm4", "hw_3", 400.0, "匝道3前"))
    ms.add_monitor(MonitoringPoint("hm5", "hw_exit", 250.0, "高速出口"))
    ms.add_monitor(MonitoringPoint("hm6", "hw_1", 100.0, "匝道1后"))
    ms.add_monitor(MonitoringPoint("hm7", "hw_2", 100.0, "匝道2后"))
    ms.add_monitor(MonitoringPoint("hm8", "hw_3", 100.0, "匝道3后"))

    engine = SimulationEngine(network, ms, dt=0.1, seed=seed)
    engine.set_driver_factory(create_driver_factory())

    engine.add_spawn_config("hw_entry", 0.5, DEFAULT_DRIVER_WEIGHTS)
    engine.add_spawn_config("on_ramp_1", 0.15, DEFAULT_DRIVER_WEIGHTS)
    engine.add_spawn_config("on_ramp_2", 0.15, DEFAULT_DRIVER_WEIGHTS)
    engine.add_spawn_config("on_ramp_3", 0.15, DEFAULT_DRIVER_WEIGHTS)

    return engine
