"""
场景4：十字立交桥
主路：东西/南北各3车道，限速100km/h
4个环形匝道(NE, SE, SW, NW)，限速40km/h
"""

from simulation.road_network import (
    RoadNetwork, RoadSegment, Intersection, Connection,
)
from simulation.monitor import MonitorSystem, MonitoringPoint
from simulation.engine import SimulationEngine
from simulation.drivers.base import create_driver_factory, DEFAULT_DRIVER_WEIGHTS


SCENARIO_NAME = "interchange"
SCENARIO_DESC = "十字立交桥(主路限速100km/h)"


def build_interchange(seed=None) -> SimulationEngine:
    network = RoadNetwork()

    hw_limit = 100.0 / 3.6   # 27.78 m/s
    ramp_limit = 40.0 / 3.6  # 11.11 m/s

    # 东西方向主路
    network.add_segment(RoadSegment("ew_w_in", 600, 3, hw_limit, "highway"))    # 西进
    network.add_segment(RoadSegment("ew_w_out", 600, 3, hw_limit, "highway"))   # 西出（立交西侧出）
    network.add_segment(RoadSegment("ew_e_in", 600, 3, hw_limit, "highway"))    # 东进
    network.add_segment(RoadSegment("ew_e_out", 600, 3, hw_limit, "highway"))   # 东出

    # 南北方向主路(立交层，高架)
    network.add_segment(RoadSegment("ns_n_in", 600, 3, hw_limit, "highway"))    # 北进
    network.add_segment(RoadSegment("ns_n_out", 600, 3, hw_limit, "highway"))   # 北出
    network.add_segment(RoadSegment("ns_s_in", 600, 3, hw_limit, "highway"))    # 南进
    network.add_segment(RoadSegment("ns_s_out", 600, 3, hw_limit, "highway"))   # 南出

    # 直行通过段（立交中心区）
    network.add_segment(RoadSegment("ew_thru", 200, 3, hw_limit, "highway"))    # 东西直行
    network.add_segment(RoadSegment("ns_thru", 200, 3, hw_limit, "highway"))    # 南北直行

    # 4个环形匝道（右转匝道）
    network.add_segment(RoadSegment("ramp_ne", 400, 1, ramp_limit, "ramp", curvature=0.8))  # 东→北
    network.add_segment(RoadSegment("ramp_se", 400, 1, ramp_limit, "ramp", curvature=0.8))  # 南→东
    network.add_segment(RoadSegment("ramp_sw", 400, 1, ramp_limit, "ramp", curvature=0.8))  # 西→南
    network.add_segment(RoadSegment("ramp_nw", 400, 1, ramp_limit, "ramp", curvature=0.8))  # 北→西

    # 分流/汇流点
    # 东西方向
    network.add_intersection(Intersection("div_w", ["ew_w_in"], ["ew_thru", "ramp_sw"]))
    network.add_intersection(Intersection("merge_e", ["ew_thru", "ramp_se"], ["ew_e_out"]))
    network.add_intersection(Intersection("div_e", ["ew_e_in"], ["ew_thru", "ramp_ne"]))  # 反向
    network.add_intersection(Intersection("merge_w", ["ew_thru", "ramp_nw"], ["ew_w_out"]))  # 反向

    # 南北方向
    network.add_intersection(Intersection("div_n", ["ns_n_in"], ["ns_thru", "ramp_nw"]))
    network.add_intersection(Intersection("merge_s", ["ns_thru", "ramp_sw"], ["ns_s_out"]))
    network.add_intersection(Intersection("div_s", ["ns_s_in"], ["ns_thru", "ramp_se"]))
    network.add_intersection(Intersection("merge_n", ["ns_thru", "ramp_ne"], ["ns_n_out"]))

    # 连接 - 直行
    network.add_connection(Connection("ew_w_in", "ew_thru", "div_w", "straight", 5.0))
    network.add_connection(Connection("ew_thru", "ew_e_out", "merge_e", "straight", 5.0))
    network.add_connection(Connection("ns_n_in", "ns_thru", "div_n", "straight", 5.0))
    network.add_connection(Connection("ns_thru", "ns_s_out", "merge_s", "straight", 5.0))

    # 连接 - 匝道分流
    network.add_connection(Connection("ew_w_in", "ramp_sw", "div_w", "diverge", 0.8))  # 西入→南出
    network.add_connection(Connection("ew_e_in", "ramp_ne", "div_e", "diverge", 0.8))  # 东入→北出
    network.add_connection(Connection("ns_n_in", "ramp_nw", "div_n", "diverge", 0.8))  # 北入→西出
    network.add_connection(Connection("ns_s_in", "ramp_se", "div_s", "diverge", 0.8))  # 南入→东出

    # 连接 - 匝道汇流
    network.add_connection(Connection("ramp_se", "ew_e_out", "merge_e", "merge", 1.0))
    network.add_connection(Connection("ramp_nw", "ew_w_out", "merge_w", "merge", 1.0))
    network.add_connection(Connection("ramp_ne", "ns_n_out", "merge_n", "merge", 1.0))
    network.add_connection(Connection("ramp_sw", "ns_s_out", "merge_s", "merge", 1.0))

    network.build_index()

    # 监测点
    ms = MonitorSystem()
    ms.add_monitor(MonitoringPoint("im1", "ew_w_in", 300.0, "西进口"))
    ms.add_monitor(MonitoringPoint("im2", "ew_e_in", 300.0, "东进口"))
    ms.add_monitor(MonitoringPoint("im3", "ns_n_in", 300.0, "北进口"))
    ms.add_monitor(MonitoringPoint("im4", "ns_s_in", 300.0, "南进口"))
    ms.add_monitor(MonitoringPoint("im5", "ew_e_out", 300.0, "东出口"))
    ms.add_monitor(MonitoringPoint("im6", "ew_w_out", 300.0, "西出口"))
    ms.add_monitor(MonitoringPoint("im7", "ns_n_out", 300.0, "北出口"))
    ms.add_monitor(MonitoringPoint("im8", "ns_s_out", 300.0, "南出口"))
    ms.add_monitor(MonitoringPoint("im9", "ramp_ne", 200.0, "东北匝道"))
    ms.add_monitor(MonitoringPoint("im10", "ramp_se", 200.0, "东南匝道"))
    ms.add_monitor(MonitoringPoint("im11", "ramp_sw", 200.0, "西南匝道"))
    ms.add_monitor(MonitoringPoint("im12", "ramp_nw", 200.0, "西北匝道"))

    engine = SimulationEngine(network, ms, dt=0.1, seed=seed)
    engine.set_driver_factory(create_driver_factory())

    engine.add_spawn_config("ew_w_in", 0.4, DEFAULT_DRIVER_WEIGHTS)
    engine.add_spawn_config("ew_e_in", 0.4, DEFAULT_DRIVER_WEIGHTS)
    engine.add_spawn_config("ns_n_in", 0.4, DEFAULT_DRIVER_WEIGHTS)
    engine.add_spawn_config("ns_s_in", 0.4, DEFAULT_DRIVER_WEIGHTS)

    return engine
