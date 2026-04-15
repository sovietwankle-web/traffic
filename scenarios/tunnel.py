"""
场景6：长隧道路段
进口段100m(限速60km/h) -> 隧道1500m(限速80km/h, 2车道, 禁止换道) -> 出口段100m(限速60km/h)
"""

from simulation.road_network import (
    RoadNetwork, RoadSegment, Intersection, Connection,
)
from simulation.monitor import MonitorSystem, MonitoringPoint
from simulation.engine import SimulationEngine
from simulation.drivers.base import create_driver_factory, DEFAULT_DRIVER_WEIGHTS


SCENARIO_NAME = "tunnel"
SCENARIO_DESC = "长隧道路段(隧道限速80km/h,禁止换道)"


def build_tunnel(seed=None) -> SimulationEngine:
    network = RoadNetwork()

    approach_limit = 60.0 / 3.6  # 16.67 m/s
    tunnel_limit = 80.0 / 3.6   # 22.22 m/s

    # 进口段
    network.add_segment(RoadSegment("approach", 200, 2, approach_limit, "urban"))
    # 隧道（分为4段以便放置监测点）
    tunnel_seg_len = 1500.0 / 4
    for i in range(4):
        network.add_segment(RoadSegment(
            f"tunnel_{i}", tunnel_seg_len, 2, tunnel_limit, "tunnel", no_lane_change=True
        ))
    # 出口段
    network.add_segment(RoadSegment("departure", 200, 2, approach_limit, "urban"))

    # 连接点
    network.add_intersection(Intersection("tunnel_entry", ["approach"], ["tunnel_0"]))
    for i in range(3):
        network.add_intersection(Intersection(f"tunnel_mid_{i}", [f"tunnel_{i}"], [f"tunnel_{i+1}"]))
    network.add_intersection(Intersection("tunnel_exit", ["tunnel_3"], ["departure"]))

    # 连接
    network.add_connection(Connection("approach", "tunnel_0", "tunnel_entry", "straight"))
    for i in range(3):
        network.add_connection(Connection(f"tunnel_{i}", f"tunnel_{i+1}", f"tunnel_mid_{i}", "straight"))
    network.add_connection(Connection("tunnel_3", "departure", "tunnel_exit", "straight"))

    network.build_index()

    # 监测点
    ms = MonitorSystem()
    ms.add_monitor(MonitoringPoint("tm1", "approach", 100.0, "隧道前方"))
    ms.add_monitor(MonitoringPoint("tm2", "tunnel_0", 187.5, "隧道入口段"))
    ms.add_monitor(MonitoringPoint("tm3", "tunnel_1", 187.5, "隧道前1/3"))
    ms.add_monitor(MonitoringPoint("tm4", "tunnel_2", 187.5, "隧道中段"))
    ms.add_monitor(MonitoringPoint("tm5", "tunnel_3", 187.5, "隧道后1/3"))
    ms.add_monitor(MonitoringPoint("tm6", "departure", 100.0, "隧道出口后"))

    engine = SimulationEngine(network, ms, dt=0.1, seed=seed)
    engine.set_driver_factory(create_driver_factory())

    engine.add_spawn_config("approach", 0.4, DEFAULT_DRIVER_WEIGHTS)

    return engine
