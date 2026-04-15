"""
场景1：两个连续的十字路口
主路：东西方向，2车道，限速50km/h
支路：南北方向，1车道，限速40km/h
两个信号灯路口，间距150m
"""

from simulation.road_network import (
    RoadNetwork, RoadSegment, Intersection, Connection, TrafficLight,
)
from simulation.monitor import MonitorSystem, MonitoringPoint
from simulation.engine import SimulationEngine
from simulation.drivers.base import create_driver_factory, DEFAULT_DRIVER_WEIGHTS


SCENARIO_NAME = "consecutive_intersections"
SCENARIO_DESC = "两个连续信号灯十字路口(主路限速50km/h)"


def build_consecutive_intersections(seed=None) -> SimulationEngine:
    network = RoadNetwork()

    # 主路路段(东西方向) - 限速50km/h = 13.89m/s
    main_limit = 50.0 / 3.6
    network.add_segment(RoadSegment("main_w", 200, 2, main_limit, "urban"))      # 西入口→路口1
    network.add_segment(RoadSegment("main_mid", 150, 2, main_limit, "urban"))    # 路口1→路口2
    network.add_segment(RoadSegment("main_e", 200, 2, main_limit, "urban"))      # 路口2→东出口

    # 支路路段(南北方向) - 限速40km/h = 11.11m/s
    side_limit = 40.0 / 3.6
    # 路口1的支路
    network.add_segment(RoadSegment("side1_n", 150, 1, side_limit, "urban"))     # 北入口→路口1
    network.add_segment(RoadSegment("side1_s", 150, 1, side_limit, "urban"))     # 路口1→南出口
    # 路口2的支路
    network.add_segment(RoadSegment("side2_n", 150, 1, side_limit, "urban"))     # 北入口→路口2
    network.add_segment(RoadSegment("side2_s", 150, 1, side_limit, "urban"))     # 路口2→南出口

    # 交通灯
    light1 = TrafficLight(
        cycle_time=60.0,
        green_ratios={"main_w": 0.45, "side1_n": 0.35},  # 主路绿灯27s，支路21s，黄灯12s
        offset=0.0,
    )
    light2 = TrafficLight(
        cycle_time=60.0,
        green_ratios={"main_mid": 0.45, "side2_n": 0.35},
        offset=10.0,  # 相位偏移，模拟绿波
    )

    # 交叉口
    int1 = Intersection("int1", ["main_w", "side1_n"], ["main_mid", "side1_s"], light1)
    int2 = Intersection("int2", ["main_mid", "side2_n"], ["main_e", "side2_s"], light2)
    network.add_intersection(int1)
    network.add_intersection(int2)

    # 连接关系 - 主路直行
    network.add_connection(Connection("main_w", "main_mid", "int1", "straight", 3.0))
    network.add_connection(Connection("main_mid", "main_e", "int2", "straight", 3.0))
    # 支路直行
    network.add_connection(Connection("side1_n", "side1_s", "int1", "straight", 1.0))
    network.add_connection(Connection("side2_n", "side2_s", "int2", "straight", 1.0))
    # 右转（主路→支路）
    network.add_connection(Connection("main_w", "side1_s", "int1", "right", 0.5))
    network.add_connection(Connection("main_mid", "side2_s", "int2", "right", 0.5))
    # 左转（支路→主路）
    network.add_connection(Connection("side1_n", "main_mid", "int1", "left", 0.5))
    network.add_connection(Connection("side2_n", "main_e", "int2", "left", 0.5))

    network.build_index()

    # 监测点
    monitor_system = MonitorSystem()
    monitor_system.add_monitor(MonitoringPoint("m1", "main_w", 100.0, "西入口主路"))
    monitor_system.add_monitor(MonitoringPoint("m2", "main_w", 190.0, "路口1西侧"))
    monitor_system.add_monitor(MonitoringPoint("m3", "main_mid", 75.0, "两路口之间"))
    monitor_system.add_monitor(MonitoringPoint("m4", "main_mid", 140.0, "路口2西侧"))
    monitor_system.add_monitor(MonitoringPoint("m5", "main_e", 50.0, "路口2东侧"))
    monitor_system.add_monitor(MonitoringPoint("m6", "main_e", 180.0, "东出口主路"))
    # 支路监测点
    monitor_system.add_monitor(MonitoringPoint("m7", "side1_n", 100.0, "路口1北侧支路"))
    monitor_system.add_monitor(MonitoringPoint("m8", "side2_n", 100.0, "路口2北侧支路"))

    # 仿真引擎
    engine = SimulationEngine(network, monitor_system, dt=0.1, seed=seed)
    engine.set_driver_factory(create_driver_factory())

    # 车辆生成
    engine.add_spawn_config("main_w", 0.3, DEFAULT_DRIVER_WEIGHTS)
    engine.add_spawn_config("side1_n", 0.1, DEFAULT_DRIVER_WEIGHTS)
    engine.add_spawn_config("side2_n", 0.1, DEFAULT_DRIVER_WEIGHTS)

    return engine
