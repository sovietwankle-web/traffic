"""
场景3：井字双向单车道胡同路段
3x3网格，每条路段80m，限速30km/h
交叉口无信号灯（让行优先）
"""

from simulation.road_network import (
    RoadNetwork, RoadSegment, Intersection, Connection,
)
from simulation.monitor import MonitorSystem, MonitoringPoint
from simulation.engine import SimulationEngine
from simulation.drivers.base import create_driver_factory, DEFAULT_DRIVER_WEIGHTS


SCENARIO_NAME = "hutong_grid"
SCENARIO_DESC = "井字双向单车道胡同(限速30km/h)"


def build_hutong_grid(seed=None) -> SimulationEngine:
    network = RoadNetwork()

    limit = 30.0 / 3.6  # 8.33 m/s

    # 3x3网格：9个节点(交叉口)，12条路段
    # 节点命名：(row, col) -> "n_r_c"
    # 路段命名：水平 "h_r_c1_c2"，垂直 "v_r1_r2_c"

    # 交叉口
    for r in range(3):
        for c in range(3):
            network.add_intersection(Intersection(f"n_{r}_{c}"))

    # 水平路段（东西方向）- 双向用两条单向路段模拟
    for r in range(3):
        for c in range(2):
            # 东向
            seg_e = RoadSegment(f"h_{r}_{c}_{c+1}", 80, 1, limit, "alley", is_bidirectional=True)
            network.add_segment(seg_e)
            # 西向
            seg_w = RoadSegment(f"h_{r}_{c+1}_{c}", 80, 1, limit, "alley", is_bidirectional=True)
            network.add_segment(seg_w)

    # 垂直路段（南北方向）
    for r in range(2):
        for c in range(3):
            # 南向
            seg_s = RoadSegment(f"v_{r}_{r+1}_{c}", 80, 1, limit, "alley", is_bidirectional=True)
            network.add_segment(seg_s)
            # 北向
            seg_n = RoadSegment(f"v_{r+1}_{r}_{c}", 80, 1, limit, "alley", is_bidirectional=True)
            network.add_segment(seg_n)

    # 入口路段（从网格边缘进入）
    entry_segs = []
    for r in range(3):
        # 西侧入口
        seg = RoadSegment(f"entry_w_{r}", 50, 1, limit, "alley")
        network.add_segment(seg)
        entry_segs.append(seg.id)
        # 东侧入口
        seg = RoadSegment(f"entry_e_{r}", 50, 1, limit, "alley")
        network.add_segment(seg)
        entry_segs.append(seg.id)
    for c in range(3):
        # 北侧入口
        seg = RoadSegment(f"entry_n_{c}", 50, 1, limit, "alley")
        network.add_segment(seg)
        entry_segs.append(seg.id)
        # 南侧入口
        seg = RoadSegment(f"entry_s_{c}", 50, 1, limit, "alley")
        network.add_segment(seg)
        entry_segs.append(seg.id)

    # 出口路段
    for r in range(3):
        network.add_segment(RoadSegment(f"exit_w_{r}", 50, 1, limit, "alley"))
        network.add_segment(RoadSegment(f"exit_e_{r}", 50, 1, limit, "alley"))
    for c in range(3):
        network.add_segment(RoadSegment(f"exit_n_{c}", 50, 1, limit, "alley"))
        network.add_segment(RoadSegment(f"exit_s_{c}", 50, 1, limit, "alley"))

    # 连接：入口→网格内部
    for r in range(3):
        network.add_connection(Connection(f"entry_w_{r}", f"h_{r}_0_1", f"n_{r}_0", "straight", 1.0))
        network.add_connection(Connection(f"entry_e_{r}", f"h_{r}_2_1", f"n_{r}_2", "straight", 1.0))
    for c in range(3):
        network.add_connection(Connection(f"entry_n_{c}", f"v_0_1_{c}", f"n_0_{c}", "straight", 1.0))
        network.add_connection(Connection(f"entry_s_{c}", f"v_2_1_{c}", f"n_2_{c}", "straight", 1.0))

    # 连接：网格内部路段间
    for r in range(3):
        for c in range(2):
            int_from = f"n_{r}_{c}"
            int_to = f"n_{r}_{c+1}"
            # 东向：到达c+1后可以继续东/转南/转北
            if c + 1 < 2:
                network.add_connection(Connection(f"h_{r}_{c}_{c+1}", f"h_{r}_{c+1}_{c+2}", int_to, "straight", 2.0))
            if r < 2:
                network.add_connection(Connection(f"h_{r}_{c}_{c+1}", f"v_{r}_{r+1}_{c+1}", int_to, "right", 1.0))
            if r > 0:
                network.add_connection(Connection(f"h_{r}_{c}_{c+1}", f"v_{r}_{r-1}_{c+1}", int_to, "left", 1.0))

            # 西向
            if c > 0:
                network.add_connection(Connection(f"h_{r}_{c+1}_{c}", f"h_{r}_{c}_{c-1}", int_from, "straight", 2.0))
            if r < 2:
                network.add_connection(Connection(f"h_{r}_{c+1}_{c}", f"v_{r}_{r+1}_{c}", int_from, "right", 1.0))
            if r > 0:
                network.add_connection(Connection(f"h_{r}_{c+1}_{c}", f"v_{r}_{r-1}_{c}", int_from, "left", 1.0))

    for r in range(2):
        for c in range(3):
            int_from = f"n_{r}_{c}"
            int_to = f"n_{r+1}_{c}"
            # 南向
            if r + 1 < 2:
                network.add_connection(Connection(f"v_{r}_{r+1}_{c}", f"v_{r+1}_{r+2}_{c}", int_to, "straight", 2.0))
            if c < 2:
                network.add_connection(Connection(f"v_{r}_{r+1}_{c}", f"h_{r+1}_{c}_{c+1}", int_to, "right", 1.0))
            if c > 0:
                network.add_connection(Connection(f"v_{r}_{r+1}_{c}", f"h_{r+1}_{c}_{c-1}", int_to, "left", 1.0))

            # 北向
            if r > 0:
                network.add_connection(Connection(f"v_{r+1}_{r}_{c}", f"v_{r}_{r-1}_{c}", int_from, "straight", 2.0))
            if c < 2:
                network.add_connection(Connection(f"v_{r+1}_{r}_{c}", f"h_{r}_{c}_{c+1}", int_from, "right", 1.0))
            if c > 0:
                network.add_connection(Connection(f"v_{r+1}_{r}_{c}", f"h_{r}_{c}_{c-1}", int_from, "left", 1.0))

    # 连接：网格内部→出口
    for r in range(3):
        # 到达最东列可以出去
        network.add_connection(Connection(f"h_{r}_1_2", f"exit_e_{r}", f"n_{r}_2", "straight", 0.5))
        # 到达最西列可以出去
        network.add_connection(Connection(f"h_{r}_1_0", f"exit_w_{r}", f"n_{r}_0", "straight", 0.5))
    for c in range(3):
        # 到达最南行可以出去
        network.add_connection(Connection(f"v_1_2_{c}", f"exit_s_{c}", f"n_2_{c}", "straight", 0.5))
        # 到达最北行可以出去
        network.add_connection(Connection(f"v_1_0_{c}", f"exit_n_{c}", f"n_0_{c}", "straight", 0.5))

    network.build_index()

    # 监测点：每个交叉口一个
    ms = MonitorSystem()
    for r in range(3):
        for c in range(3):
            # 在通向该交叉口的某条路段上放置
            if c > 0:
                seg_id = f"h_{r}_{c-1}_{c}"
                ms.add_monitor(MonitoringPoint(f"hm_{r}_{c}", seg_id, 60.0, f"胡同路口({r},{c})"))
            elif r > 0:
                seg_id = f"v_{r-1}_{r}_{c}"
                ms.add_monitor(MonitoringPoint(f"hm_{r}_{c}", seg_id, 60.0, f"胡同路口({r},{c})"))
            else:
                seg_id = f"entry_w_0"
                ms.add_monitor(MonitoringPoint(f"hm_{r}_{c}", seg_id, 30.0, f"胡同路口({r},{c})"))

    engine = SimulationEngine(network, ms, dt=0.1, seed=seed)
    engine.set_driver_factory(create_driver_factory())

    # 从各个边缘入口生成车辆
    for seg_id in entry_segs:
        engine.add_spawn_config(seg_id, 0.05, DEFAULT_DRIVER_WEIGHTS)

    return engine
