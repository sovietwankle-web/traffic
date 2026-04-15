"""
场景7：事故导致4车道→2车道→4车道
正常段4车道100km/h → 过渡段(200m) → 施工段2车道60km/h(800m) → 过渡段(200m) → 恢复4车道100km/h
"""

from simulation.road_network import (
    RoadNetwork, RoadSegment, Intersection, Connection,
)
from simulation.monitor import MonitorSystem, MonitoringPoint
from simulation.engine import SimulationEngine
from simulation.drivers.base import create_driver_factory, DEFAULT_DRIVER_WEIGHTS


SCENARIO_NAME = "bottleneck"
SCENARIO_DESC = "事故瓶颈:4车道变2车道再变4车道(施工段限速60km/h)"


def build_bottleneck(seed=None) -> SimulationEngine:
    network = RoadNetwork()

    normal_limit = 100.0 / 3.6   # 27.78 m/s
    taper_limit = 80.0 / 3.6     # 22.22 m/s
    work_limit = 60.0 / 3.6      # 16.67 m/s

    # 正常段→过渡段→施工段→过渡段→正常段
    network.add_segment(RoadSegment("normal_1", 500, 4, normal_limit, "highway"))
    network.add_segment(RoadSegment("taper_1", 200, 3, taper_limit, "highway"))    # 4→2过渡
    network.add_segment(RoadSegment("work_1", 400, 2, work_limit, "highway"))      # 施工段前半
    network.add_segment(RoadSegment("work_2", 400, 2, work_limit, "highway"))      # 施工段后半
    network.add_segment(RoadSegment("taper_2", 200, 3, taper_limit, "highway"))    # 2→4过渡
    network.add_segment(RoadSegment("normal_2", 500, 4, normal_limit, "highway"))

    # 连接
    network.add_intersection(Intersection("merge_point", ["normal_1"], ["taper_1"]))
    network.add_intersection(Intersection("narrow_start", ["taper_1"], ["work_1"]))
    network.add_intersection(Intersection("work_mid", ["work_1"], ["work_2"]))
    network.add_intersection(Intersection("narrow_end", ["work_2"], ["taper_2"]))
    network.add_intersection(Intersection("expand_point", ["taper_2"], ["normal_2"]))

    network.add_connection(Connection("normal_1", "taper_1", "merge_point", "straight"))
    network.add_connection(Connection("taper_1", "work_1", "narrow_start", "merge"))
    network.add_connection(Connection("work_1", "work_2", "work_mid", "straight"))
    network.add_connection(Connection("work_2", "taper_2", "narrow_end", "straight"))
    network.add_connection(Connection("taper_2", "normal_2", "expand_point", "diverge"))

    network.build_index()

    # 监测点
    ms = MonitorSystem()
    ms.add_monitor(MonitoringPoint("bm1", "normal_1", 250.0, "正常段入口"))
    ms.add_monitor(MonitoringPoint("bm2", "normal_1", 450.0, "汇流前"))
    ms.add_monitor(MonitoringPoint("bm3", "taper_1", 100.0, "过渡段"))
    ms.add_monitor(MonitoringPoint("bm4", "work_1", 200.0, "施工段前部"))
    ms.add_monitor(MonitoringPoint("bm5", "work_2", 200.0, "施工段后部"))
    ms.add_monitor(MonitoringPoint("bm6", "taper_2", 100.0, "恢复过渡段"))
    ms.add_monitor(MonitoringPoint("bm7", "normal_2", 250.0, "恢复正常段"))
    ms.add_monitor(MonitoringPoint("bm8", "normal_2", 450.0, "正常段出口"))

    engine = SimulationEngine(network, ms, dt=0.1, seed=seed)
    engine.set_driver_factory(create_driver_factory())

    engine.add_spawn_config("normal_1", 0.5, DEFAULT_DRIVER_WEIGHTS)

    return engine
