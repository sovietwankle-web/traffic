"""
道路网络模型：有向图表示的路段、交叉口、连接关系。
车辆在路段上沿连续坐标移动(0 ~ segment.length)。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RoadSegment:
    """道路路段"""
    id: str
    length: float               # 路段长度(m)
    num_lanes: int              # 车道数
    speed_limit: float          # 限速(m/s)
    segment_type: str = "urban" # urban/highway/tunnel/ramp/alley
    curvature: float = 0.0     # 曲率(0=直线)
    is_bidirectional: bool = False  # 是否双向(胡同场景)
    no_lane_change: bool = False   # 是否禁止换道(隧道)


@dataclass
class TrafficLight:
    """交通灯"""
    cycle_time: float           # 总周期(s)
    green_ratios: Dict[str, float]  # segment_id -> 绿灯占比
    offset: float = 0.0        # 相位偏移(s)，用于协调多个路口

    def is_green(self, segment_id: str, current_time: float) -> bool:
        if segment_id not in self.green_ratios:
            return True
        phase = (current_time + self.offset) % self.cycle_time
        ratio = self.green_ratios[segment_id]
        # 按segment在green_ratios中的顺序分配相位
        cumulative = 0.0
        for sid, r in self.green_ratios.items():
            if sid == segment_id:
                return cumulative <= phase < cumulative + r * self.cycle_time
            cumulative += r * self.cycle_time
        return False


@dataclass
class Intersection:
    """交叉口"""
    id: str
    incoming_segments: List[str] = field(default_factory=list)
    outgoing_segments: List[str] = field(default_factory=list)
    traffic_light: Optional[TrafficLight] = None

    def is_green_for(self, segment_id: str, current_time: float) -> bool:
        if self.traffic_light is None:
            return True  # 无信号灯，始终可通行
        return self.traffic_light.is_green(segment_id, current_time)


@dataclass
class Connection:
    """路段间的连接关系"""
    from_segment: str
    to_segment: str
    through_intersection: Optional[str] = None
    turn_type: str = "straight"  # straight/left/right/merge/diverge
    weight: float = 1.0         # 路径选择权重


class RoadNetwork:
    """道路网络：有向图"""

    def __init__(self):
        self.segments: Dict[str, RoadSegment] = {}
        self.intersections: Dict[str, Intersection] = {}
        self.connections: List[Connection] = []
        # 预计算邻接关系
        self._outgoing: Dict[str, List[Connection]] = {}  # from_seg -> connections
        self._incoming: Dict[str, List[Connection]] = {}   # to_seg -> connections
        # 路段到下游交叉口的映射
        self._segment_end_intersection: Dict[str, Optional[str]] = {}

    def add_segment(self, segment: RoadSegment):
        self.segments[segment.id] = segment

    def add_intersection(self, intersection: Intersection):
        self.intersections[intersection.id] = intersection

    def add_connection(self, connection: Connection):
        self.connections.append(connection)
        self._outgoing.setdefault(connection.from_segment, []).append(connection)
        self._incoming.setdefault(connection.to_segment, []).append(connection)
        # 更新路段末端交叉口映射
        if connection.through_intersection:
            self._segment_end_intersection[connection.from_segment] = connection.through_intersection

    def get_outgoing_connections(self, segment_id: str) -> List[Connection]:
        return self._outgoing.get(segment_id, [])

    def get_incoming_connections(self, segment_id: str) -> List[Connection]:
        return self._incoming.get(segment_id, [])

    def get_segment(self, segment_id: str) -> Optional[RoadSegment]:
        return self.segments.get(segment_id)

    def get_intersection_at_end(self, segment_id: str) -> Optional[Intersection]:
        iid = self._segment_end_intersection.get(segment_id)
        if iid:
            return self.intersections.get(iid)
        return None

    def get_entry_segments(self) -> List[str]:
        """获取入口路段（没有上游连接的路段）"""
        all_to = {c.to_segment for c in self.connections}
        return [sid for sid in self.segments if sid not in all_to]

    def get_exit_segments(self) -> List[str]:
        """获取出口路段（没有下游连接的路段）"""
        all_from = {c.from_segment for c in self.connections}
        return [sid for sid in self.segments if sid not in all_from]

    def build_index(self):
        """重建索引（在批量添加后调用）"""
        self._outgoing.clear()
        self._incoming.clear()
        self._segment_end_intersection.clear()
        for c in self.connections:
            self._outgoing.setdefault(c.from_segment, []).append(c)
            self._incoming.setdefault(c.to_segment, []).append(c)
            if c.through_intersection:
                self._segment_end_intersection[c.from_segment] = c.through_intersection
